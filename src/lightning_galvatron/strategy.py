import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from galvatron.pipeline import PipelineParallel
from lightning_utilities import module_available
from megatron import get_args
from megatron.initialize import initialize_megatron
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.utilities.seed import reset_seed
    from lightning.pytorch import Trainer
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.ddp import DDPStrategy
    from lightning.pytorch.trainer.states import TrainerFn
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.utilities.seed import reset_seed
    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.trainer.states import TrainerFn
    from pytorch_lightning.utilities.exceptions import MisconfigurationException
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class GalvatronStrategy(DDPStrategy):
    """Galvatron training strategy for Pytorch Lightning.

    .. warning:: ``GalvatronStrategy`` is in beta and subject to change.

    Galvatron provides a solution to layer-wise hybrid parallel training for Transformer models. It supports
    the mixture of 4 types of parallelisms including data parallel, sharded data parallel, tensor parallel,
    and pipeline parallel.

    Arguments:
        model_type: The model type that Galvatron supports for hybrid parallelism, selected from ["chatglm", "gpt"].
            More models are coming soon.
            (Default: "chatglm")
        hp_config: The path of hybrid parallel config. If specified, Galvatron will use this file to configure
            hybrid parallel strategies; otherwise, other arguments for parallelism take effect.
            (Default: None)
        model_size: Model size, selected from ["gpt-1.5b", "gpt-2.7b", "gpt-6.7b"] when model_type is "gpt".
            If not spedified, Galvatron will use other arguments to determine the model hyper-parameters.
            (Default: None)
        global_train_batch_size: Global training batch size, taking effect when `hp_config` is None.
            (Default: 32)
        seq_length: Input sequence length, taking effect when `model_size` is None.
            (Default: 128)
        num_attention_heads: Number of attention heads, taking effect when `model_size` is None.
            (Default: 16)
        hidden_size: Hidden size of Transformer model, taking effect when `model_size` is None.
            (Default: 1024)
        num_hidden_layers: Number of hidden layers, taking effect when `model_size` is None.
            (Default: 28)
        pp_deg: Pipeline parallel degree, taking effect when `hp_config` is None. `num_hidden_layers` and `world_size`
            should be divisible by it.
            (Default: 1)
        global_tp_deg: Global tensor parallel degree, taking effect when `hp_config` is None. `num_attention_heads`
            should be divisible by it.
            (Default: 1)
        fsdp: Apply FSDP for all transformer layers, taking effect when `hp_config` is None. Selected from [0, 1].
            (Default: 0)
        global_checkpoint: Wrap all layers with PyTorch checkpoint wrapper, taking effect when `hp_config` is None.
            Selected from [0, 1].
            (Default: 0)
        mixed_precision: Mixed precision option, selected from ["fp32", "fp16", "bf16"]
            (Default: "bf16")
        pipeline_type: Galvatron pipeline type, selected from ["gpipe", "pipedream_flush"]
            (Default: "gpipe")
        default_dp_type: Default data parallel type, selected from ["ddp", "zero2", "zero3"]
            (Default: "zero2")
        use_flash_attn: Use flash attention to optimise attention calculation.
            (Default: False)
    """

    strategy_name = "galvatron"

    def __init__(
        self,
        model_type: str = "chatglm",
        hp_config: str = None,
        model_size: str = None,
        global_train_batch_size: int = 32,
        seq_length: int = 128,
        num_attention_heads: int = 16,
        hidden_size: int = 1024,
        num_hidden_layers: int = 28,
        pp_deg: int = 1,
        global_tp_deg: int = 1,
        chunks: int = -1,
        global_tp_consec: int = 1,
        fsdp: int = 0,
        global_checkpoint: int = 0,
        mixed_precision: str = "bf16",
        pipeline_type: str = "gpipe",
        default_dp_type: str = "zero2",
        use_flash_attn: bool = False,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        **galvatron_kwargs: Union[Any, Dict[str, Any]],
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        assert model_type in ["gpt", "chatglm"]
        self._model_type = model_type

        if model_size is not None:
            assert model_size in [
                "gpt-1.5b",
                "gpt-2.7b",
                "gpt-6.7b",
            ]
        assert mixed_precision in ["fp32", "fp16", "bf16"]
        assert pipeline_type in ["gpipe", "pipedream_flush"]
        assert default_dp_type in ["ddp", "zero2"]

        self._galvatron_kwargs = galvatron_kwargs

        self._args_defaults = {
            "global_train_batch_size": global_train_batch_size,
            "seq_length": seq_length,
            "num_attention_heads": num_attention_heads,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "pp_deg": pp_deg,
            "global_tp_deg": global_tp_deg,
            "chunks": chunks,
            "global_tp_consec": global_tp_consec,
            "fsdp": fsdp,
            "global_checkpoint": global_checkpoint,
            "mixed_precision": mixed_precision,
            "pipeline_type": pipeline_type,
            "default_dp_type": default_dp_type,
            "galvatron_config_path": hp_config,
            "gradient_accumulation_fusion": False,
            "async_tensor_model_parallel_allreduce": False,
            "use_flash_attn": use_flash_attn,
            "openai_gelu": True,
            "initialize_on_meta": False,
        }

    def setup_distributed(self) -> None:
        reset_seed()

        self.set_world_ranks()

        self._init_galvatron_distributed()

    def _init_galvatron_distributed(self) -> None:
        self._set_node_environment_variables()

        torch.cuda.set_device(self.local_rank)

    def _set_node_environment_variables(self) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def setup(self, trainer: Trainer) -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING:
            # Galvatron: do distributed environment initialization and megatron args preparation
            self._args, _ = initialize_megatron(args_defaults=self._args_defaults, ignore_unknown_args=True)
            self._args = get_args()

            # Galvatron: configure the model, make it ready for pipeline parallel,
            # tensor parallel and (sharded) data parallel
            self._configure_galvatron_model(self._args)

            self.setup_optimizers(trainer)

    def _configure_galvatron_model(self, args: Any) -> None:
        assert self.model.model is not None
        model = self.model.model
        self.model.model = self._setup_model(model, args)

    def _setup_model(self, model: Module, args: Any) -> PipelineParallel:
        if self._model_type == "gpt":
            from galvatron.gpt.hybrid_parallel_model_dist import (
                construct_hybrid_parallel_model,
                get_hybrid_parallel_configs,
                overwrite_megatron_args,
            )
        elif self._model_type == "chatglm":
            from galvatron.chatglm.hybrid_parallel_model_dist import (
                construct_hybrid_parallel_model,
                get_hybrid_parallel_configs,
                overwrite_megatron_args,
            )
        else:
            raise MisconfigurationException

        # Galvatron: overwrite model configs and megatron args, contruct the hybrid parallel form of the model
        config = model.config
        overwrite_megatron_args(config, args)
        hybrid_parallel_configs = get_hybrid_parallel_configs(args)
        model = construct_hybrid_parallel_model(
            model=model, model_config=model.config, training_args=args, hybrid_parallel_configs=hybrid_parallel_configs
        )
        return model

    def training_step(self, *args: Any, **kwargs: Any) -> Dict:
        assert self.model is not None and self.model.model is not None
        model = self.model.model

        batch, batch_idx = args

        # Galvatron: prepare the batched inputs
        if self._model_type == "gpt":
            from galvatron.gpt.train_hp_layerwise_dist import forward_step_func

            input_ids = batch
            batch = [[input_ids], [input_ids]]
        elif self._model_type == "chatglm":
            from galvatron.chatglm.train_hp_layerwise_dist import forward_step_func

            input_ids, labels = batch
            batch = [[input_ids], [labels]]
        else:
            raise MisconfigurationException

        # Galvatron: do pipeline parallel
        if self._args.pipeline_type == "gpipe":
            loss = model.gpipe_forward(forward_step_func, batch)
            model.gpipe_backward()
        elif self._args.pipeline_type == "pipedream_flush":
            loss = model.pipedream_flush_forward_backward(forward_step_func, batch)
        else:
            raise MisconfigurationException

        avg_loss = torch.tensor(np.mean([l_.item() for l_ in loss])) if len(loss) else torch.tensor(0)
        return {"loss": avg_loss, "loss_reduced": loss}

    # backward is included in training_step in Galvatron strategy
    def backward(self, loss: Tensor, optimizer: Optional[Optimizer]) -> Tensor:
        optimizer.step()
        return loss

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def post_training_step(self) -> None:
        pass
