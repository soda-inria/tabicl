from collections import OrderedDict

import pytest
import torch

from tabicl._model.inference import InferenceManager
from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import skip_if_device_unusable


def test_icl_predictions_leave_their_input_unchanged():
    """ICL must not add the labels into row representations in-place.

    When a forward pass raises ``torch.OutOfMemoryError``, ``InferenceManager`` halves
    the batch size and runs the forward function again on the same inputs, so the
    forward function must leave its inputs unchanged.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/185.
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
    """An OOM when all inputs fit in one batch is retried with half the batch size.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/179.
    """
    skip_if_device_unusable(device)
    mgr = InferenceManager(enc_name="tf_icl", out_dim=3)
    mgr.configure(device=device)
    batch_sizes = []

    def forward(x):
        batch_sizes.append(x.shape[0])
        if len(batch_sizes) == 1:
            raise torch.OutOfMemoryError("simulated")
        return x * 2

    x = torch.randn(4, 8, 3, device=device)
    out = mgr(forward, OrderedDict(x=x))
    assert batch_sizes == [4, 2, 2]
    assert torch.equal(out, x * 2)


@pytest.mark.parametrize("device", ["cuda", "xpu"])
@pytest.mark.parametrize("n_rows", [1, 4])
def test_oom_at_minimum_batch_size_mentions_host_ram(monkeypatch, device, n_rows):
    """On a GPU that may share host RAM, the error at the minimum batch size says so.

    One row fails in the single-batch path, four rows in the batched loop.
    """
    skip_if_device_unusable(device)
    mgr = InferenceManager(enc_name="tf_icl", out_dim=3)
    mgr.configure(device=device)
    monkeypatch.setattr(mgr, "_unified_host", True)

    def forward(x):
        raise torch.OutOfMemoryError("simulated")

    with pytest.raises(RuntimeError, match="minimum batch size.*allocates from host RAM"):
        mgr(forward, OrderedDict(x=torch.zeros(n_rows, 8, 3, device=device)))
