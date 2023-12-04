# Lightning integration with Galvatron

[Galvatron](https://www.vldb.org/pvldb/vol16/p470-miao.pdf) is a new system framework aiming at efficient Transformer training over multiple GPUs using automatic parallelism.

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI - Version](https://img.shields.io/pypi/v/hetu-galvatron)](https://pypi.org/project/hetu-galvatron/)
[![CI testing](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-galvatron/badge/?version=latest)](https://lightning-galvatron.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-Universe/lightning-Galvatron/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-Universe/lightning-Galvatron/main)

It incorporates multiple popular parallelism dimensions (including data parallel, sharded data parallel, tensor parallel and pipeline parallel) and automatically finds the most efficient hybrid parallelism strategy.

Galvatron can be configured in the training script by specifying strategy arguments as follows:

```py
from lightning import Trainer
from lightning_galvatron import GalvatronStrategy

trainer = Trainer(
    strategy=GalvatronStrategy(
        model_type="gpt",
        model_size="gpt-1.5b",
        pp_deg=2,
        global_tp_deg=2,
        fsdp=1
    ),
    accelerator="gpu",
    devices=4
)
```

Please see the official [Galvatron repository](https://github.com/Hsword/Hetu/tree/main/tools/Galvatron) for more model support and advanced features.

## Cite

If you use Galvatron in a scientific publication, we would appreciate citations to the following paper:

```
@article{miao2023galvatron,
  title = {Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism},
  author = {Miao, Xupeng and Wang, Yujie and Jiang, Youhe and Shi, Chunan and Nie, Xiaonan and Zhang, Hailin and Cui, Bin},
  journal = {Proc. {VLDB} Endow.},
  volume = {16},
  number = {3},
  pages = {470--479},
  year = {2023},
  doi = {10.14778/3570690.3570697},
  publisher = {VLDB Endowment},
}
```
