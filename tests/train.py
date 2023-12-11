"""This script is meant to be executed from `tests/helper.py::_run_galvatron`.

Because Galvatron uses megatron to initialize the distribution environment, and we cannot find a way to completely clean
the state between tests using only one process.
"""

import argparse
import json
import os
from collections import namedtuple

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, DistributedSampler

from lightning_galvatron import GalvatronStrategy

_PATH_HERE = os.path.dirname(__file__)

try:
    from flash_attn.models.gpt import GPTLMHeadModel
    from galvatron.gpt.dataloader import DataLoaderForGPT
    from galvatron.gpt.gpt_config_utils import gpt_config, overwrite_configs_and_args

    class GPTModel(LightningModule):
        def __init__(self, strategy_options):
            super().__init__()
            self.global_train_batch_size = strategy_options["global_train_batch_size"]
            self.pp_deg = strategy_options["pp_deg"]

            Args = namedtuple(
                "Args",
                [
                    "overwrite_config",
                    "model_size",
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                    "seq_length",
                    "use_flash_attn",
                ],
            )
            self.args = Args(
                overwrite_config=True,
                model_size=strategy_options["model_size"],
                vocab_size=50257,
                hidden_size=strategy_options["hidden_size"],
                num_hidden_layers=strategy_options["num_hidden_layers"],
                seq_length=strategy_options["seq_length"],
                use_flash_attn=strategy_options["use_flash_attn"],
            )
            self.config = gpt_config(self.args)
            overwrite_configs_and_args(self.config, self.args)

            self.model = GPTLMHeadModel(self.config, device="cpu")

        def training_step(self, batch, batch_idx):
            input_ids = batch
            output = self.model(input_ids)
            return output

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=0.01)
            return optimizer

        def train_dataloader(self) -> TRAIN_DATALOADERS:
            dataset = DataLoaderForGPT(self.args, dataset_size=self.global_train_batch_size * 20)
            data_num_replicas = self.trainer.world_size // self.pp_deg
            train_batch_size_input = self.global_train_batch_size // data_num_replicas
            trainloader = DataLoader(
                dataset=dataset,
                batch_size=train_batch_size_input,
                sampler=DistributedSampler(
                    dataset,
                    shuffle=True,
                    num_replicas=data_num_replicas,
                    rank=self.trainer.local_rank % data_num_replicas,
                ),
            )
            return trainloader

except Exception:
    gpt_config, overwrite_configs_and_args, DataLoaderForGPT, GPTLMHeadModel = None, None, None, None
    GPTModel = None


from galvatron.chatglm.dataloader import DataLoaderForChatGLM  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402


class ChatGLMModel(LightningModule):
    def __init__(self, strategy_options):
        super().__init__()
        self.global_train_batch_size = strategy_options["global_train_batch_size"]
        self.pp_deg = strategy_options["pp_deg"]

        self.config = AutoConfig.from_pretrained(
            os.path.join(_PATH_HERE, os.pardir, "models", "chatglm"), trust_remote_code=True
        )
        # self.config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        Args = namedtuple(
            "Args", ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "seq_length"]
        )
        self.args = Args(
            vocab_size=self.config.vocab_size,
            hidden_size=strategy_options["hidden_size"],
            num_hidden_layers=strategy_options["num_hidden_layers"],
            num_attention_heads=strategy_options["num_attention_heads"],
            seq_length=strategy_options["seq_length"],
        )
        self.config.hidden_size = self.args.hidden_size
        self.config.inner_hidden_size = self.args.hidden_size * 4
        self.config.num_layers = self.args.num_hidden_layers
        self.config.num_attention_heads = self.args.num_attention_heads
        self.config.max_sequence_length = self.args.seq_length

        self.model = AutoModel.from_config(self.config, trust_remote_code=True)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        output = self.model(input_ids, labels)
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = DataLoaderForChatGLM(self.args, self.config, dataset_size=self.global_train_batch_size * 20)
        data_num_replicas = self.trainer.world_size // self.pp_deg
        train_batch_size_input = self.global_train_batch_size // data_num_replicas
        trainloader = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size_input,
            sampler=DistributedSampler(
                dataset, shuffle=True, num_replicas=data_num_replicas, rank=self.trainer.local_rank % data_num_replicas
            ),
        )
        return trainloader


parser = argparse.ArgumentParser()
parser.add_argument("--trainer-options", required=True)
parser.add_argument("--strategy-options", required=True)


def run_test_from_config(trainer_options, strategy_options):
    if strategy_options["model_type"] == "chatglm":
        model = ChatGLMModel(strategy_options)
    elif strategy_options["model_type"] == "gpt":
        model = GPTModel(strategy_options)
    else:
        raise MisconfigurationException

    trainer = Trainer(strategy=GalvatronStrategy(**strategy_options), **trainer_options)
    trainer.fit(model)


if __name__ == "__main__":
    args = parser.parse_args()
    run_test_from_config(
        json.loads(args.trainer_options),
        json.loads(args.strategy_options),
    )
