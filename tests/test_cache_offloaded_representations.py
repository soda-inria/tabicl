"""Caching must work when the row representations were offloaded from the device.

The row-wise interaction returns its output where the offload settings of
``ROW_CONFIG`` put it, which can be CPU memory while the model runs on an
accelerator. Storing a cache then adds the label embedding, computed on the
device, to those representations. The model is small and randomly initialized.
"""

from __future__ import annotations

import pytest
import torch

from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import DEVICES, skip_if_device_unusable

TRAIN_SIZE = 20
TEST_SIZE = 10
N_FEATURES = 5


def _make_model(max_classes: int) -> TabICL:
    torch.manual_seed(0)
    model = TabICL(
        max_classes=max_classes,
        num_quantiles=9,
        embed_dim=16,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
    )
    return model.eval()


def _config(device: str, row_offload: str) -> InferenceConfig:
    mgr = {"device": device, "use_amp": False, "use_fa3": False, "offload": "gpu"}
    return InferenceConfig(COL_CONFIG=dict(mgr), ROW_CONFIG={**mgr, "offload": row_offload}, ICL_CONFIG=dict(mgr))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cache_mode", ["kv", "repr"])
@pytest.mark.parametrize("max_classes, num_classes", [(4, 3), (0, 0)], ids=["classification", "regression"])
def test_cache_with_offloaded_representations(device, cache_mode, max_classes, num_classes):
    """Storing and using a cache from CPU-offloaded representations matches keeping them on the device."""
    skip_if_device_unusable(device)
    model = _make_model(max_classes).to(device)
    gen = torch.Generator().manual_seed(0)
    X_train = torch.randn(2, TRAIN_SIZE, N_FEATURES, generator=gen).to(device)
    X_test = torch.randn(2, TEST_SIZE, N_FEATURES, generator=gen).to(device)
    if num_classes > 0:
        y_train = torch.randint(0, num_classes, (2, TRAIN_SIZE), generator=gen).float()
        y_train[:, :num_classes] = torch.arange(num_classes).float()  # every table contains every class
    else:
        y_train = torch.randn(2, TRAIN_SIZE, generator=gen)
    y_train = y_train.to(device)

    def run(row_offload: str):
        config = _config(device, row_offload)
        stored = model.forward_with_cache(
            X_train=X_train, y_train=y_train, X_test=X_test, cache_mode=cache_mode, inference_config=config
        )
        used = model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False, inference_config=config)
        return stored, used

    with torch.no_grad():
        expected_stored, expected_used = run("gpu")
        stored, used = run("cpu")

    torch.testing.assert_close(stored.cpu(), expected_stored.cpu())
    torch.testing.assert_close(used.cpu(), expected_used.cpu())
