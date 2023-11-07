# Lightning integration with Galvatron

[Galvatron]() is a new system framework aiming at efficient Transformer training over multiple GPUs using automatic parallelism.

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

Please see the official [Galvatron repository]() for more model support and advanced features.

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-galvatron/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-galvatron/badge/?version=latest)](https://lightning-galvatron.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-galvatron/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-galvatron/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [ ] lock the main breach in GH setting - no direct push without PR
- [ ] init Read-The-Docs (add this new project)
- [ ] add credentials for releasing package to PyPI

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
