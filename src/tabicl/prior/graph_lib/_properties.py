from typing import List

import numpy as np

from tabicl.prior.graph_lib._base import Context


def sample_categorical_sizes(n_features: int, context: Context, max_cat_size: int = 200) -> List[int]:
    """Sample categorical sizes for features."""

    cat_fraction = np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0)
    n_cat = round(n_features * cat_fraction)
    n_cont = n_features - n_cat
    # Use the provided context to ensure device consistency

    n_local = context.sampler.randint("cat_sizes_n_local", 0, n_cat + 1, mode="local", boundary_mass=True)
    n_meta = n_cat - n_local
    if context.config.use_corrected_cat_sizes:
        cat_size_limit = context.sampler.randint("cat_size_limit", 2, max_cat_size + 1, mode="local", use_log=True)
    else:
        # originally used
        cat_size_limit = context.sampler.randint("cat_size_limit", 2, min(10, max_cat_size + 1), use_log=True)
    # only allow large categories in "local" mode, otherwise runtimes for some datasets can get very large
    cat_sizes = (
        [0] * n_cont
        + [
            context.sampler.randint("cat_size", 2, cat_size_limit + 1, use_log=True, mode="local")
            for _ in range(n_local)
        ]
        + [
            context.sampler.randint("cat_size", 2, min(10, cat_size_limit + 1), use_log=True, mode="meta")
            for _ in range(n_meta)
        ]
    )

    return cat_sizes
