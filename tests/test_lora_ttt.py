from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Lora_ttt.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_lora_ttt_module():
    spec = importlib.util.spec_from_file_location("lora_ttt_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEncoder(torch.nn.Module):
    def __init__(self, lora_ttt, num_blocks: int = 2):
        super().__init__()
        attention_cls = lora_ttt._get_lora_attention_class()
        self.blocks = torch.nn.ModuleList(
            [FakeTransformerBlock(attention_cls) for _ in range(num_blocks)]
        )


class FakeTransformerBlock(torch.nn.Module):
    def __init__(self, attention_cls):
        super().__init__()
        self.attn = attention_cls(embed_dim=4, num_heads=2, dropout=0.0)
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 4)


class FakeRowInteractor(torch.nn.Module):
    def __init__(self, lora_ttt):
        super().__init__()
        self.tf_row = FakeEncoder(lora_ttt)


class FakeIclPredictor(torch.nn.Module):
    def __init__(self, lora_ttt):
        super().__init__()
        self.tf_icl = FakeEncoder(lora_ttt)


class FakeTabICL(torch.nn.Module):
    def __init__(self, lora_ttt):
        super().__init__()
        self.col_embedder = torch.nn.Linear(4, 4)
        self.row_interactor = FakeRowInteractor(lora_ttt)
        self.icl_predictor = FakeIclPredictor(lora_ttt)


class FakeClassifier:
    def __init__(self, model):
        self.model_ = model


def test_row_icl_lora_only_trains_lora_params_and_freezes_col_embedder():
    lora_ttt = load_lora_ttt_module()
    model = FakeTabICL(lora_ttt)
    classifier = FakeClassifier(model)

    trainable_params = lora_ttt._configure_ttt_trainable_params(
        classifier,
        lora_ttt.TTTConfig(
            enabled=True,
            lora=True,
            lora_targets="row,icl",
            lora_rank=2,
            lora_alpha=4.0,
        ),
    )

    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    assert trainable_params
    assert trainable_names
    assert all("._ttt_lora_" in name for name in trainable_names)
    assert not any(param.requires_grad for param in model.col_embedder.parameters())
    assert model.row_interactor.tf_row.blocks[0].attn._ttt_lora_in_proj_installed is True
    assert model.icl_predictor.tf_icl.blocks[0].attn._ttt_lora_in_proj_installed is True


def test_merged_ttt_state_dict_excludes_lora_keys_and_merges_linear_delta():
    lora_ttt = load_lora_ttt_module()
    model = FakeTabICL(lora_ttt)
    classifier = FakeClassifier(model)

    lora_ttt._configure_ttt_trainable_params(
        classifier,
        lora_ttt.TTTConfig(
            enabled=True,
            lora=True,
            lora_targets="row,icl",
            lora_rank=1,
            lora_alpha=1.0,
        ),
    )

    linear = model.row_interactor.tf_row.blocks[0].linear1
    base_weight = linear.weight.detach().clone()
    with torch.no_grad():
        getattr(linear, lora_ttt._lora_param_name("A")).fill_(1.0)
        getattr(linear, lora_ttt._lora_param_name("B")).fill_(2.0)

    merged = lora_ttt._merged_ttt_state_dict(model)

    assert not any("_ttt_lora_" in name for name in merged)
    merged_weight = merged["row_interactor.tf_row.blocks.0.linear1.weight"]
    expected_weight = base_weight + torch.full_like(base_weight, 2.0)
    torch.testing.assert_close(merged_weight, expected_weight)
