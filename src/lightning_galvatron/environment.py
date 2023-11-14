import logging
import os
import torch

from lightning.fabric.plugins import ClusterEnvironment

log = logging.getLogger(__name__)


class GalvatronEnvironment(ClusterEnvironment):
    """Environment for distributed training with `Galvatron`."""

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    def main_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", "6000"))

    @property
    def service_port(self) -> int:
        return int(os.environ.get("GALVATRON_SERVICE_PORT", -1))

    @staticmethod
    def detect() -> bool:
        return "GALVATRON_SERVICE_PORT" in os.environ

    def world_size(self) -> int:
        return torch.distributed.get_world_size()

    def set_world_size(self, size: int) -> None:
        log.debug("`GalvatronEnvironment.set_world_size` was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return torch.distributed.get_rank()

    def set_global_rank(self, rank: int) -> None:
        log.debug("`GalvatronEnvironment.set_global_rank` was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        return int(os.environ.get("NODE_RANK", 0))