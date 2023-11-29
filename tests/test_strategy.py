import os

import pytest
import torch

from tests.helper import _run_galvatron

try:
    import flash_attn
except ImportError:
    flash_attn = None

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_chatglm_with_json_config(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 2,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "chatglm",
        "hp_config": str(os.path.join(_PATH_HERE, "configs", "galvatron_config_chatglm_2gpus_example.json")),
        "global_train_batch_size": 4,
        "pp_deg": 1,
        "global_tp_deg": 1,
        "seq_length": 128,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "pipeline_type": "pipedream_flush",
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
def test_chatglm_with_pipeline_parallel(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "chatglm",
        "global_train_batch_size": 8,
        "pp_deg": 2,
        "global_tp_deg": 1,
        "seq_length": 128,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "pipeline_type": "pipedream_flush",
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
def test_chatglm_with_tensor_parallel(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "chatglm",
        "global_train_batch_size": 8,
        "pp_deg": 1,
        "global_tp_deg": 2,
        "seq_length": 128,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "pipeline_type": "pipedream_flush",
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
@pytest.mark.skipif(flash_attn is None, reason="This test needs flash-attn.")
def test_chatglm_with_flash_attn(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "chatglm",
        "global_train_batch_size": 8,
        "pp_deg": 2,
        "global_tp_deg": 1,
        "seq_length": 128,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "pipeline_type": "pipedream_flush",
        "use_flash_attn": True,
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
@pytest.mark.skipif(flash_attn is None, reason="This test needs flash-attn.")
def test_gpt_with_hybrid_parallel(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "gpt",
        "model_size": "gpt-1.5b",
        "global_train_batch_size": 8,
        "pp_deg": 2,
        "global_tp_deg": 2,
        "seq_length": 512,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "pipeline_type": "pipedream_flush",
        "use_flash_attn": True,
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
@pytest.mark.skipif(flash_attn is None, reason="This test needs flash-attn.")
def test_gpt_with_gpipe(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "gpt",
        "model_size": "gpt-1.5b",
        "global_train_batch_size": 8,
        "pp_deg": 2,
        "global_tp_deg": 2,
        "seq_length": 512,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "pipeline_type": "gpipe",
        "use_flash_attn": True,
    }
    _run_galvatron(trainer_options, strategy_options)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test needs at least 4 GPUs.")
@pytest.mark.skipif(flash_attn is None, reason="This test needs flash-attn.")
def test_gpt_with_other_configurations(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 4,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    strategy_options = {
        "model_type": "gpt",
        "model_size": "gpt-1.5b",
        "global_train_batch_size": 8,
        "pp_deg": 2,
        "global_tp_deg": 2,
        "seq_length": 512,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "global_tp_consec": 0,
        "fsdp": 1,
        "global_checkpoint": 1,
        "mixed_precision": "bf16",
        "pipeline_type": "gpipe",
        "default_dp_type": "ddp",
        "use_flash_attn": True,
    }
    _run_galvatron(trainer_options, strategy_options)
