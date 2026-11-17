import torch

from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL


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
