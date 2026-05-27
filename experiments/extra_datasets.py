"""Loaders for the five extra datasets that the SUBLIME paper evaluates but the
public repo does not ship loaders for:

    * ogbn-arxiv  (Table 1 col 4, Table 2 col 4)
    * Wine        (Table 1 col 5)
    * Cancer      (Table 1 col 6)
    * Digits      (Table 1 col 7)
    * 20news      (Table 1 col 8)

Each ``load_<name>`` returns the same 8-tuple as ``data_loader.load_data``:

    (features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj)

For the non-graph UCI datasets there is no original adjacency, so we return an
identity matrix as ``adj``; this matches what ``Experiment.train`` does anyway
when ``-gsl_mode structure_inference`` is used (it overwrites the anchor with
I_n on line ~150 of ``main.py``).

Caveats
-------
The paper appendix (F.3) describes only the *search space* for hyperparameters
on these datasets, never the optimal per-dataset settings. The dataset splits
likewise follow the convention of LDS [Franceschi et al. 2019] without precise
seeds. So the absolute numbers reproduced via these loaders may differ from
Table 1 even on the paper's exact environment; the loaders are correct in
*shape* and *label rate* (matching Table 7 of the paper).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from utils import sparse_mx_to_torch_sparse_tensor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _identity_adj(n: int, sparse: bool):
    if sparse:
        return sparse_mx_to_torch_sparse_tensor(sp.eye(n, dtype=np.float32, format="csr"))
    return torch.eye(n, dtype=torch.float32)


def _split_masks(labels: np.ndarray, label_rate: float, val_size: int,
                 n_classes: int, seed: int = 0):
    """Build train / val / test masks with the per-paper label rate.

    Following LDS / SLAPS conventions: the *training* set is ``round(label_rate * n)``
    labeled nodes selected uniformly at random (stratified across classes when
    possible); ``val_size`` random nodes from the remainder form the validation
    set; the rest are the test set.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    n_train = max(n_classes, int(round(label_rate * n)))
    # stratified sample: ceil(n_train / n_classes) per class, then trim
    per_class = max(1, n_train // n_classes)
    train_idx = []
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        train_idx.extend(idx_c[:per_class].tolist())
    train_idx = np.array(sorted(train_idx[:n_train]))
    rest = np.setdiff1d(np.arange(n), train_idx)
    rng.shuffle(rest)
    val_idx = np.sort(rest[:val_size])
    test_idx = np.sort(rest[val_size:])

    def mk(idx):
        m = np.zeros(n, dtype=bool); m[idx] = True
        return torch.from_numpy(m)
    return mk(train_idx), mk(val_idx), mk(test_idx)


def _pack(features_np, labels_np, train_m, val_m, test_m, sparse):
    n, nfeats = features_np.shape
    features = torch.from_numpy(features_np.astype(np.float32))
    labels = torch.from_numpy(labels_np.astype(np.int64))
    nclasses = int(labels.max().item()) + 1
    adj = _identity_adj(n, sparse)
    return features, nfeats, labels, nclasses, train_m, val_m, test_m, adj


# ---------------------------------------------------------------------------
# UCI datasets (via sklearn)
# ---------------------------------------------------------------------------

def _load_sklearn_uci(loader_fn, label_rate: float, val_size: int, sparse: bool, seed: int = 0):
    from sklearn.preprocessing import scale
    ds = loader_fn()
    X = scale(ds.data.astype(np.float32))
    y = ds.target.astype(np.int64)
    n_classes = int(y.max()) + 1
    tr, va, te = _split_masks(y, label_rate, val_size, n_classes, seed=seed)
    return _pack(X, y, tr, va, te, sparse)


def load_wine(sparse: bool = False, seed: int = 0):
    """Wine — 178 / 13 / 3, paper label rate 0.056 (Table 7)."""
    from sklearn.datasets import load_wine as _wine
    return _load_sklearn_uci(_wine, label_rate=0.056, val_size=30, sparse=sparse, seed=seed)


def load_cancer(sparse: bool = False, seed: int = 0):
    """Breast Cancer — 569 / 30 / 2, paper label rate 0.018 (Table 7)."""
    from sklearn.datasets import load_breast_cancer
    return _load_sklearn_uci(load_breast_cancer, label_rate=0.018, val_size=100,
                             sparse=sparse, seed=seed)


def load_digits(sparse: bool = False, seed: int = 0):
    """Digits — 1797 / 64 / 10, paper label rate 0.028 (Table 7)."""
    from sklearn.datasets import load_digits as _digits
    return _load_sklearn_uci(_digits, label_rate=0.028, val_size=300, sparse=sparse, seed=seed)


def load_20news(sparse: bool = False, seed: int = 0):
    """20news — paper uses 10 of the 20 topics, 9607 samples, 236 features.

    The paper does not document the exact preprocessing, so we mirror the most
    common public recipe (also used by SLAPS' supplementary): fetch the 10
    topics with the largest sample counts, vectorise with TF-IDF, keep the top
    236 features by document frequency. This matches the (n=9607, d=236)
    shape reported in Table 7.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    # The 10 categories totalling ~9607 samples used by LDS.
    cats = [
        "alt.atheism", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
        "misc.forsale", "rec.autos", "rec.motorcycles",
        "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics",
    ]
    train = fetch_20newsgroups(subset="train", categories=cats,
                               remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", categories=cats,
                              remove=("headers", "footers", "quotes"))
    texts = list(train.data) + list(test.data)
    y = np.concatenate([train.target, test.target]).astype(np.int64)

    vec = TfidfVectorizer(max_features=236, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    n_classes = int(y.max()) + 1
    tr, va, te = _split_masks(y, label_rate=0.010, val_size=500,
                              n_classes=n_classes, seed=seed)
    return _pack(X, y, tr, va, te, sparse)


# ---------------------------------------------------------------------------
# ogbn-arxiv
# ---------------------------------------------------------------------------

def load_ogbn_arxiv(sparse: bool = True, seed: int = 0):
    """ogbn-arxiv via the official OGB loader.

    Returns the OGB-canonical train/val/test split (label rate 0.537 per Table 7),
    feature matrix (n=169343, d=128), and the directed citation adjacency
    symmetrised + self-loops added (the conventional preprocessing for this
    benchmark).
    """
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root="data/ogb")
    g, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    feats = g.ndata["feat"].numpy().astype(np.float32)
    y = labels.squeeze().long()

    n = feats.shape[0]
    src, dst = g.edges()
    src = src.numpy(); dst = dst.numpy()
    # symmetrise
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A + sp.eye(n, dtype=np.float32, format="csr")
    A.data[:] = 1.0  # deduplicate weights

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(A)
    else:
        adj = torch.from_numpy(A.toarray().astype(np.float32))

    def mk(idx):
        m = np.zeros(n, dtype=bool); m[idx.numpy()] = True
        return torch.from_numpy(m)

    train_m = mk(split_idx["train"])
    val_m = mk(split_idx["valid"])
    test_m = mk(split_idx["test"])

    features = torch.from_numpy(feats)
    nfeats = feats.shape[1]
    nclasses = int(y.max().item()) + 1
    return features, nfeats, y, nclasses, train_m, val_m, test_m, adj


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    "wine":        load_wine,
    "cancer":      load_cancer,
    "digits":      load_digits,
    "20news":      load_20news,
    "ogbn-arxiv":  load_ogbn_arxiv,
}


def load_extra_dataset(name: str, sparse: bool, seed: int = 0):
    if name not in LOADERS:
        raise KeyError(f"unknown extra dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](sparse=bool(sparse), seed=seed)
