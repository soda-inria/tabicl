# The TabICLv2 Prior

The TabICLv2 prior is divided into samplers for different types of objects: 
datasets, functions, graphs, matrices, and so on.
Many sampler types are implemented as base classes with subclasses,
where the base class (e.g., RandomFunction) samples a random subclass 
(e.g., RandomGPFunction),
and then the subclass implements a specific mechanism for sampling.

One can sample a dataset as follows:
```python
from tabicl.prior.graph_lib.dataset import RandomDataset
from tabicl.prior.graph_lib.base import Context, DatasetProperties

# sample dataset with 1000 total samples 
# (only n_train + n_test matters, there is no split currently)
# get 2 numerical features (cat size 0), 
# one categorical with (up to) 4 different values,
# and one target with (up to) 3 different classes.
ds_prop = DatasetProperties(n_train=1000, 
                            n_test=0, 
                            cat_sizes={"x": [0, 0, 4], "y": [3]})
tensors = RandomDataset(Context()).sample(ds_prop).get_concat_tensors()
x_num = tensors['x_num']  # (n_samples, n_features)
x_cat = tensors['x_cat']
y_cat = tensors['y_cat']
```

However, this is **not the full dataset sampling logic**: 
Preprocessing, filtering, and some hyperparameter sampling are done in
`prior/dataset.py` and `prior/graph_scm.py`.

## Essential classes
- `PriorConfig` (`config.py`) contains configuration options for the prior. 
(This includes options to fix some unintended behavior 
that was present during TabICLv2 pre-training; 
we set the defaults to retain the TabICLv2 behavior.)
- `GlobalSampler` (`base.py`) allows sampling 
scalar variables from different distributions, 
with different correlation modes:
  - `'global'` samples a single value per name, 
  - `'meta'` samples correlated values for the same name,
  - `'local'` samples independent values for the same name.
- `Context` (`base.py`): Stores a `PriorConfig` and a `GlobalSampler`.
Gets passed to every sampling class.
- `PriorComponent` (`base.py`): Base class for other sampling classes.
Takes a `Context` object and uses it to track potential 
for infinite recursions.
- `RandomTransformer` (`base.py`) and subclasses: 
Implement a simple fit-predict interface for things like random functions
such that they can be applied to multiple tensors 
while only being fitted on the first one. 
Currently, we only apply them once, 
so the fit-transform paradigm is not necessary.

`base.py` also contains classes to 
store and specify datasets and their properties.

## Dataset sampling hierarchy:
A random dataset (`dataset.py`) is sampled 
by sampling a random graph (`graph.py`), assigning features to nodes, and
evaluating the graph using a random graph function (`graph_function.py`).
The graph function evaluates the nodes 
using random node functions (`node_function.py`).
The node function applies different processing steps, 
including converters for extracting dataset features (`converter.py`),
sampling random points (`points.py`) on root nodes, 
and applying random multi-functions (`multi_function.py`) on other nodes. 
The multi-functions use aggregation mechanisms 
with random functions (`function.py`), 
which can use random matrices (`matrix.py`), 
random activations (`activation.py`),
and random weights (`weights.py`).

In addition, `properties.py` provides a way to sample categorical sizes, 
which can be passed to `RandomDataset`.


