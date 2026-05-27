"""
Helper runner for Figure 5 (robustness on Cora).

Randomly perturbs the original Cora adjacency by either deleting or adding edges
at a given rate, then runs SUBLIME in structure-refinement mode on the perturbed
graph. Reports best test accuracy averaged over ``-ntrials`` seeds, matching the
SUBLIME curve in Figure 5(a)/(b).

Usage (from repo root via the env wrapper):

    sublime-python -m experiments.robustness_runner \\
        -dataset cora -perturb delete -rate 0.5 -ntrials 5

Notes:
    * Only ``-dataset cora`` is supported (Figure 5 in the paper is Cora-only).
    * ``-perturb`` is one of ``delete`` or ``add``; ``-rate`` is in [0, 0.95].
    * Hyperparameters default to ``scripts/cora_sr.sh``; override on the CLI.
    * Baselines (GCN, Pro-GNN) are reported in the paper as static numbers and
      are not retrained here.
"""

import argparse
import sys

import numpy as np
import scipy.sparse as sp
import torch

import main as sublime_main
import data_loader as sublime_data_loader
from utils import sparse_mx_to_torch_sparse_tensor


def _perturb_adj_dense(adj: np.ndarray, rate: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of the (symmetric, no self-loops) Cora adjacency with a
    fraction ``rate`` of edges either dropped (``mode='delete'``) or extra random
    non-edges added (``mode='add'``).

    The perturbation is symmetric and self-loops are never introduced.
    """
    n = adj.shape[0]
    # work on upper triangle to keep symmetry tractable
    iu, ju = np.triu_indices(n, k=1)
    edge_mask = adj[iu, ju] > 0
    edges = np.stack([iu[edge_mask], ju[edge_mask]], axis=1)
    m = edges.shape[0]
    new = adj.copy()

    if mode == "delete":
        n_drop = int(round(rate * m))
        if n_drop > 0:
            drop_idx = rng.choice(m, size=n_drop, replace=False)
            for i, j in edges[drop_idx]:
                new[i, j] = 0.0
                new[j, i] = 0.0
    elif mode == "add":
        n_add = int(round(rate * m))
        if n_add > 0:
            # candidate non-edges (upper triangle, no self-loop, currently 0)
            cand_mask = ~edge_mask
            cand_i = iu[cand_mask]
            cand_j = ju[cand_mask]
            c = cand_i.shape[0]
            add_idx = rng.choice(c, size=min(n_add, c), replace=False)
            for i, j in zip(cand_i[add_idx], cand_j[add_idx]):
                new[i, j] = 1.0
                new[j, i] = 1.0
    else:
        raise ValueError(f"unknown perturb mode {mode!r}")
    return new


def _build_perturbed_load_data(rate: float, mode: str, seed: int):
    """Return a drop-in replacement for ``data_loader.load_data`` that perturbs
    the Cora adjacency before returning it.
    """
    rng = np.random.default_rng(seed)
    original = sublime_data_loader.load_data

    def patched(args):
        out = original(args)
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj = out
        # adj is dense float tensor (because sparse=0 for cora) or sparse torch
        if isinstance(adj, torch.Tensor) and not adj.is_sparse:
            adj_np = adj.cpu().numpy().astype(np.float32)
            new_np = _perturb_adj_dense(adj_np, rate, mode, rng)
            new_adj = torch.from_numpy(new_np)
        elif sp.issparse(adj) or (isinstance(adj, torch.Tensor) and adj.is_sparse):
            # densify, perturb, re-sparsify
            if isinstance(adj, torch.Tensor):
                dense = adj.to_dense().cpu().numpy().astype(np.float32)
            else:
                dense = adj.toarray().astype(np.float32)
            new_np = _perturb_adj_dense(dense, rate, mode, rng)
            new_adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(new_np))
        else:
            raise TypeError(f"unexpected adj type {type(adj)}")
        return (features, nfeats, labels, nclasses,
                train_mask, val_mask, test_mask, new_adj)

    return patched


def main():
    p = argparse.ArgumentParser()
    # robustness-specific
    p.add_argument("-perturb", type=str, required=True, choices=["delete", "add"])
    p.add_argument("-rate", type=float, required=True)
    p.add_argument("-perturb_seed", type=int, default=0)
    # standard SUBLIME flags (subset that we actually need; defaults match cora_sr.sh)
    p.add_argument("-ntrials", type=int, default=5)
    p.add_argument("-sparse", type=int, default=0)
    p.add_argument("-gsl_mode", type=str, default="structure_refinement")
    p.add_argument("-eval_freq", type=int, default=50)
    p.add_argument("-downstream_task", type=str, default="classification")
    p.add_argument("-dataset", type=str, default="cora")
    p.add_argument("-epochs", type=int, default=4000)
    p.add_argument("-lr", type=float, default=0.01)
    p.add_argument("-w_decay", type=float, default=0.0)
    p.add_argument("-hidden_dim", type=int, default=512)
    p.add_argument("-rep_dim", type=int, default=256)
    p.add_argument("-proj_dim", type=int, default=256)
    p.add_argument("-dropout", type=float, default=0.5)
    p.add_argument("-contrast_batch_size", type=int, default=0)
    p.add_argument("-nlayers", type=int, default=2)
    p.add_argument("-maskfeat_rate_learner", type=float, default=0.7)
    p.add_argument("-maskfeat_rate_anchor", type=float, default=0.6)
    p.add_argument("-dropedge_rate", type=float, default=0.5)
    p.add_argument("-type_learner", type=str, default="fgp")
    p.add_argument("-k", type=int, default=30)
    p.add_argument("-sim_function", type=str, default="cosine")
    p.add_argument("-gamma", type=float, default=0.9)
    p.add_argument("-activation_learner", type=str, default="relu")
    p.add_argument("-epochs_cls", type=int, default=200)
    p.add_argument("-lr_cls", type=float, default=0.001)
    p.add_argument("-w_decay_cls", type=float, default=0.0005)
    p.add_argument("-hidden_dim_cls", type=int, default=32)
    p.add_argument("-dropout_cls", type=float, default=0.5)
    p.add_argument("-dropedge_cls", type=float, default=0.75)
    p.add_argument("-nlayers_cls", type=int, default=2)
    p.add_argument("-patience_cls", type=int, default=10)
    p.add_argument("-tau", type=float, default=0.9999)
    p.add_argument("-c", type=int, default=0)
    p.add_argument("-gpu", type=int, default=0)

    args = p.parse_args()

    if args.dataset != "cora":
        print("warning: Figure 5 in the paper is Cora-only; running anyway.",
              file=sys.stderr)

    sublime_main.args = args
    sublime_main.load_data = _build_perturbed_load_data(
        args.rate, args.perturb, args.perturb_seed
    )
    print(
        f"ROBUSTNESS_CONFIG dataset={args.dataset} perturb={args.perturb} "
        f"rate={args.rate} seed={args.perturb_seed} ntrials={args.ntrials}",
        flush=True,
    )
    sublime_main.Experiment().train(args)


if __name__ == "__main__":
    main()
