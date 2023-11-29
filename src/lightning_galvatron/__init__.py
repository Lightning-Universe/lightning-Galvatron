"""Root package info."""

import os

from lightning_galvatron.__about__ import *  # noqa: F401,
from lightning_galvatron.environment import GalvatronEnvironment
from lightning_galvatron.strategy import GalvatronStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

__all__ = ["GalvatronStrategy", "GalvatronEnvironment"]
