from collections import OrderedDict

import pytest
import torch

from tabicl._model.inference import InferenceManager
from tests.torch_devices_helpers import skip_if_device_unusable


@pytest.mark.parametrize("device", ["cuda", "xpu", "mps"])
def test_oom_in_single_batch_retries_with_smaller_batches(device):
    """An OOM when all inputs fit in one batch falls back to smaller batches instead of failing.

    When a forward pass raises ``torch.OutOfMemoryError``, ``InferenceManager`` halves
    the batch size and runs the forward function again. On integrated GPUs this is how
    inference stays within the allocator cap, so the retry must also happen when all
    inputs first fit in a single batch.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/179.
    """
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
