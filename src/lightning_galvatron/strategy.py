import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

import torch
from lightning_utilities.core.imports import module_available
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.utilities.seed import reset_seed
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.ddp import DDPStrategy
    from lightning.pytorch.trainer.states import TrainerFn
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning") and module_available("lightning_fabric"):
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.utilities.seed import reset_seed
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.trainer.states import TrainerFn
    from lightning_fabric.utilities.exceptions import MisconfigurationException
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

import galvatron.site_package # to import the modified megatron in galvatron lib
from megatron.initialize import initialize_megatron
from megatron import get_args
from galvatron.pipeline import PipelineParallel

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class GalvatronStrategy(DDPStrategy):
    """Galvatron training strategy for Pytorch Lightning."""

    strategy_name = "galvatron"

    def __init__(
        self,
        model_type,
        hp_config=None,
        apply_strategy=False,
        model_size=None,
        global_train_batch_size=32,
        seq_length=128,
        num_attention_heads=16,
        hidden_size=1024,
        num_hidden_layers=28,
        pp_deg=1,
        global_tp_deg=1,
        chunks=-1,
        global_tp_consec=1,
        fsdp=0,
        global_checkpoint=0,
        mixed_precision='bf16',
        pipeline_type='gpipe',
        default_dp_type='zero2',
        use_flash_attn=False,
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

        assert model_type in ['gpt', 'chatglm']
        self._model_type = model_type

        if model_size is not None:
            assert model_size in [
                'gpt-1.5b', 'gpt-2.7b', 'gpt-6.7b',
            ]
        assert mixed_precision in ['fp32', 'fp16', 'bf16']
        assert pipeline_type in ['gpipe', 'pipedream_flush']
        assert default_dp_type in ['ddp', 'zero2']

        self._galvatron_kwargs = galvatron_kwargs

        self._args_defaults = {
            'global_train_batch_size': global_train_batch_size,
            'seq_length':seq_length,
            'num_attention_heads': num_attention_heads,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'pp_deg': pp_deg,
            'global_tp_deg': global_tp_deg,
            'chunks': chunks,
            'global_tp_consec': global_tp_consec,
            'fsdp': fsdp,
            'global_checkpoint': global_checkpoint,
            'mixed_precision': mixed_precision,
            'pipeline_type': pipeline_type,
            'default_dp_type': default_dp_type,
            'apply_strategy': apply_strategy,
            'galvatron_config_path': hp_config,
            'gradient_accumulation_fusion': False,
            'async_tensor_model_parallel_allreduce': False,
            'use_flash_attn': use_flash_attn,
            'openai_gelu': True,
            'initialize_on_meta': False
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
            self._args, _ = initialize_megatron(args_defaults=self._args_defaults, ignore_unknown_args=True)
            self._args = get_args()
            
            self._configure_galvatron_model(self._args)

            self.setup_optimizers(trainer)

    def _configure_galvatron_model(self, args) -> None:
        assert self.model.model is not None
        model = self.model.model
        self.model.model = self._setup_model(model, args)
    
    def _setup_model(self, model: Module, args) -> PipelineParallel:
        if self._model_type == 'gpt':
            from galvatron.gpt.hybrid_parallel_model_dist import overwrite_megatron_args, get_hybrid_parallel_configs, construct_hybrid_parallel_model
        elif self._model_type == 'chatglm':
            from galvatron.chatglm.hybrid_parallel_model_dist import overwrite_megatron_args, get_hybrid_parallel_configs, construct_hybrid_parallel_model
        else:
            raise MisconfigurationException
        
        config = model.config
        overwrite_megatron_args(config, args)
        hybrid_parallel_configs = get_hybrid_parallel_configs(args)
        model = construct_hybrid_parallel_model(
            model=model,
            model_config=model.config,
            training_args=args,
            hybrid_parallel_configs=hybrid_parallel_configs
        )
        return model
    
    def training_step(self, *args: Any, **kwargs: Any) -> Dict:
        assert self.model is not None and self.model.model is not None
        model = self.model.model

        batch, batch_idx = args
        
        if self._model_type == 'gpt':
            from galvatron.gpt.train_hp_layerwise_dist import forward_step_func
            input_ids = batch
            batch = [[input_ids], [input_ids]]
        elif self._model_type == 'chatglm':
            from galvatron.chatglm.train_hp_layerwise_dist import forward_step_func
            input_ids, labels = batch
            batch = [[input_ids], [labels]]
        else:
            raise MisconfigurationException
        
        if self._args.pipeline_type == 'gpipe':
            loss = model.gpipe_forward(forward_step_func, batch)
            model.gpipe_backward()
        elif self._args.pipeline_type == 'pipedream_flush':
            loss = model.pipedream_flush_forward_backward(forward_step_func, batch)
        else:
            raise MisconfigurationException
        
        if len(loss):
            avg_loss = torch.tensor(np.mean([l.item() for l in loss]))
        else:
            avg_loss = torch.tensor(0)
        return {'loss': avg_loss, 'loss_reduced': loss}
    
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
