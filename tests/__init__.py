"""Cofigure local testing."""
from lightning_utilities import module_available

if not module_available("galvatron"):
    raise ModuleNotFoundError("Galvatron is not installed!")
