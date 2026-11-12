from collections import OrderedDict

import pytest
import torch

from tabicl._model.encoders import Encoder
from tabicl._model.inference import InferenceManager


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_query_chunked_encoder_matches_full_attention(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available on this host")

    for norm_first in (True, False):
        torch.manual_seed(0)
        encoder = Encoder(
            num_blocks=2,
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            norm_first=norm_first,
            ssmax="qassmax-mlp-elementwise",
            zero_init=False,
        ).eval().to(device)
        src = torch.randn(2, 13, 16, device=device)

        with torch.no_grad():
            expected = encoder(src.clone(), train_size=9)
            actual = encoder.forward_query_chunked(src.clone(), train_size=9, chunk_size=3)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def test_cpu_manager_streams_exact_chunks_to_disk(tmp_path):
    manager = InferenceManager(enc_name="tf_col", out_dim=4)
    manager.configure(
        device="cpu",
        cpu_memory_budget_mb=0.001,
        offload="disk",
        disk_offload_dir=str(tmp_path),
        use_amp=False,
        use_fa3=False,
    )
    values = torch.randn(2, 5, 7, 4)

    def forward_fn(x):
        return x.square() + 1

    actual = manager(
        forward_fn,
        OrderedDict([("x", values)]),
    )
    expected = forward_fn(values)

    torch.testing.assert_close(actual, expected)
