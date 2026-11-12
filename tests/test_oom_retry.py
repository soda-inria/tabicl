"""Recovery from out-of-memory errors during batched inference.

When a forward pass raises ``torch.OutOfMemoryError``, ``InferenceManager`` halves
the batch size and runs the forward function again on the same inputs. On
integrated GPUs this is how inference stays within the allocator cap, so the
retry must happen in every case and must see unchanged inputs.
"""

from collections import OrderedDict

import pytest
import torch

from tabicl._model.inference import InferenceManager
from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import skip_if_device_unusable


def test_icl_predictions_leave_their_input_unchanged():
    """A retry reruns ICL on the same row representations, so ICL must not add the labels into them.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/179.
    """
    icl = TabICL().eval().icl_predictor
    torch.manual_seed(0)
    R = torch.randn(2, 20, icl.decoder[0].in_features)
    y_train = torch.randint(0, 2, (2, 15))
    R_before = R.clone()

    with torch.no_grad():
        icl._icl_predictions(R, y_train)
        icl._icl_predictions_with_cache(R, KVCache(), y_train=y_train, store_cache=True)
    assert torch.equal(R, R_before)


@pytest.mark.parametrize("device", ["cuda", "xpu", "mps"])
def test_oom_in_single_batch_retries_with_smaller_batches(device):
    """An OOM when all inputs fit in one batch falls back to smaller batches instead of failing."""
    skip_if_device_unusable(device)
    mgr = InferenceManager(enc_name="tf_icl", out_dim=3)
    mgr.configure(device=device, use_amp=False, use_fa3=False)
    batch_sizes = []

    def forward(x):
        batch_sizes.append(x.shape[0])
        if len(batch_sizes) == 1:
            raise torch.OutOfMemoryError("simulated")
        return x * 2

    x = torch.randn(4, 8, 3, device=device)
    out = mgr(forward, OrderedDict(x=x))
    assert batch_sizes == [4, 2, 2]
    torch.testing.assert_close(out, x * 2)
