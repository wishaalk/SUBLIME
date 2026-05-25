# Towards Unsupervised Deep Graph Structure Learning

This is the source code of WWW-2022 paper "[Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/pdf/2201.06367.pdf)" (SUBLIME). 

![The proposed framework](pipeline.png)

## REQUIREMENTS
This code requires the following:
* Python==3.7
* PyTorch==1.7.1
* DGL==0.7.1
* Numpy==1.20.2
* Scipy==1.6.3
* Scikit-learn==0.24.2
* Munkres==1.1.4
* ogb==1.3.1

## RUN ON KAGGLE

Open [`sublime_kaggle.ipynb`](sublime_kaggle.ipynb) on Kaggle (Settings → Accelerator → **GPU T4 x2** or **GPU P100**) and run the cells top to bottom. The notebook installs the exact paper environment (Python 3.8, PyTorch 1.7.1+cu110, DGL 0.7.1+cu110, numpy 1.20.2, scipy 1.6.3, scikit-learn 0.24.2) via micromamba + official pip wheels, then runs each experiment with the hyperparameters from `scripts/*.sh`.

> ⚠ Do **not** select an L4/A100 accelerator — PyTorch 1.7.1's cu110 wheel was not compiled for those compute capabilities.

## USAGE (local)
### Step 1: All the scripts are included in the "scripts" folder. Please get into this folder first.
```
cd scripts
```

### Step 2: Run the experiments you want:

\[Cora\]Node classification @ structure inference:
```
bash cora_si.sh
```
\[Cora\]Node classification @ structure refinement:
```
bash cora_sr.sh
```
\[Cora\]Node clustering @ structure refinement:
```
bash cora_clu.sh
```
\[Citeseer\]Node classification @ structure inference:
```
bash citeseer_si.sh
```
\[Citeseer\]Node classification @ structure refinement:
```
bash citeseer_sr.sh
```
\[Citeseer\]Node clustering @ structure refinement:
```
bash citeseer_clu.sh
```
\[Pubmed\]Node classification @ structure inference:
```
bash pubmed_si.sh
```
\[Pubmed\]Node classification @ structure refinement:
```
bash pubmed_sr.sh
```

## Cite

If you compare with, build on, or use aspects of SUBLIME framework, please cite the following:
```
@inproceedings{liu2022towards,
  title={Towards unsupervised deep graph structure learning},
  author={Liu, Yixin and Zheng, Yu and Zhang, Daokun and Chen, Hongxu and Peng, Hao and Pan, Shirui},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1392--1403},
  year={2022}
}
```
