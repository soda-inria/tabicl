"""Storing a repr cache must work when the row representations were offloaded.

The row-wise interaction returns its output where the offload settings of
``ROW_CONFIG`` put it, which can be CPU memory while the model runs on an
accelerator. Storing a repr cache then adds the label embedding, computed on
the device, to those representations.
"""

import pytest
import torch

from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import DEVICES, skip_if_device_unusable


@pytest.mark.parametrize("device", DEVICES)
def test_repr_cache_from_offloaded_representations(device):
    """Predictions match those with the representations kept on the device."""
    skip_if_device_unusable(device)
    torch.manual_seed(0)
    model = TabICL().eval().to(device)
    X_train = torch.randn(1, 20, 5, device=device)
    y_train = torch.randint(0, 3, (1, 20), device=device).float()
    X_test = torch.randn(1, 10, 5, device=device)

    def predict(row_offload):
        mgr = {"device": device, "use_amp": False}
        config = InferenceConfig(COL_CONFIG=mgr, ROW_CONFIG={**mgr, "offload": row_offload}, ICL_CONFIG=mgr)
        with torch.no_grad():
            return model.forward_with_cache(
                X_train=X_train, y_train=y_train, X_test=X_test, cache_mode="repr", inference_config=config
            )

    torch.testing.assert_close(predict("cpu"), predict("gpu"))
