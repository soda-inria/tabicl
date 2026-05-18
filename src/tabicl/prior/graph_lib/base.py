import collections
import copy
from typing import Any, Union, Dict, Optional, Literal, List, Tuple
import torch
import numpy as np

from tabiclv2.prior.graph_lib.config import PriorConfig


class GlobalSampler:
    """
    Class for sampling scalar variables (numerical, categorical) from different distributions.
    Usually, one instance of this class is shared between components of the prior,
    such that the class can correlate samples between multiple calls when using the sampling mode "meta".
    This only applies whenever the same mode is used.
    The "meta" sampling will sample variables with the same name from a distribution
    whose hyperparameters are sampled once per name. Sampling once can be realized using the sampling mode "global",
    while "local" will sample variables independently of previous calls.
    """
    def __init__(self, seed: Optional[int] = 0, config: Optional[PriorConfig] = None):
        self.globals = dict()
        self.rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(np.iinfo(np.int32).max))
        self.config = config or PriorConfig()

    def numerical(
        self,
        name: str,
        low: float,
        high: float,
        use_log: bool = False,
        use_int: bool = False,
        mode: Literal["meta", "global", "local"] = "meta",  # "meta",
        boundary_mass: bool = False,
        size: Optional[Union[int, Tuple[int]]] = None,
    ):
        # have three modes: global, meta, and local. global reduces to local.
        #  meta reduces to sampling a global meta-distribution, which then reduces to local
        assert high >= low

        # todo: maybe check if other params are equal for the stored value, or just append them to the name
        if mode == "global":
            if name not in self.globals:
                self.globals[name] = self.numerical(
                    name + "__global", low, high, use_log, use_int, mode="local", boundary_mass=boundary_mass, size=size
                )
            return self.globals[name]
        elif mode == "meta":
            if self.config.meta_sampling_mode != 'meta':
                return self.numerical(name, low, high, use_log, use_int, self.config.meta_sampling_mode, boundary_mass, size)
            loc = self.numerical(name + "__meta_beta-loc", 0.0, 1.0, mode="global", boundary_mass=False)
            sum = self.numerical(
                name + "__meta_beta-sum", 0.1, 10000.0, use_log=True, mode="global", boundary_mass=False
            )
            alpha = loc * sum
            beta = (1 - loc) * sum
            value = self.rng.beta(alpha, beta, size=size)
        else:
            value = self.rng.random(size=size)

        if use_log:
            assert low > 0
            low = np.log(low)
            high = np.log(high)

        if boundary_mass:
            value = np.clip(1.2 * value - 0.1, 0.0, 1.0)

        value = low + (high - low) * value

        if use_log:
            value = np.exp(value)
        if use_int:
            # todo: this is inconsistent with randint()
            value = np.rint(value).astype(int)
        return value

    def randint(
        self,
        name: str,
        low: int,
        high: int,
        use_log: bool = False,
        mode: Literal["meta", "global", "local"] = "meta",
        boundary_mass: bool = False,
    ) -> int:
        return np.clip(
            np.floor(self.numerical(name, low, high, use_log=use_log, mode=mode, boundary_mass=boundary_mass)).astype(
                int
            ),
            low,
            high - 1,
        )

    def categorical(self, name: str, n_categories: int, mode: Literal["meta", "global", "local"] = "meta") -> int:
        # maybe sample a vector from a GP and take the argmax? could be a bit expensive, though
        if mode == "global":
            if name not in self.globals:
                self.globals[name] = self.categorical(name + "__global", n_categories, mode="local")
            return self.globals[name]
        elif mode == "meta":
            if self.config.meta_sampling_mode != 'meta':
                return self.categorical(name, n_categories, self.config.meta_sampling_mode)
            # use get_sparse_random_weights()  (which should then not recurse onto this function!)
            ext_name = name + "__meta_weights"
            # backward compatibility because this used to be wrong in TabICLv2
            query_name = ext_name if self.config.use_corrected_cat_meta_sampling else name
            if query_name not in self.globals:
                from tabiclv2.prior.graph_lib.weights import SimpleRandomWeights

                self.globals[ext_name] = SimpleRandomWeights(Context()).sample(1, n_categories).squeeze(0)
                # self.globals[ext_name] = torch.ones(n_categories) / n_categories
                self.globals[ext_name] /= self.globals[ext_name].sum()

            return np.random.choice(np.arange(n_categories), p=self.globals[ext_name].numpy())

            # samples = torch.multinomial(self.globals[ext_name], num_samples=1)
            # # print(f'{samples=}')
            # return samples.item()

        return np.random.randint(n_categories)

    def choice(self, name: str, categories: List[Any], mode: Literal["meta", "global", "local"] = "meta") -> Any:
        return categories[self.categorical(name, len(categories), mode)]

    def boolean(self, name: str, mode: Literal["meta", "global", "local"] = "meta") -> bool:
        return self.categorical(name, 2, mode) == 1


class Context:
    """
    Class that stores the context info needed in the prior: the sampler, and the hyperparameter config.
    It also tracks for cases where a class eventually recurses onto itself, which could lead to infinite recursion.
    This happens since each such class, derived from PriorComponent, should create a derived context object
    that keeps track of which class created it.
    """
    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",  # currently only cpu works
        sampler: Optional[GlobalSampler] = None,
        config: Optional[PriorConfig] = None,
        observed_classes: Optional[List[Any]] = None,
        seed: Optional[int] = None,
    ):
        self.device = device
        self.sampler = sampler or GlobalSampler(seed=seed, config=config)
        self.config = config or PriorConfig()
        self.observed_classes = observed_classes or []
        # could also add statistics for how often which class is used

    def get_recursion_context(self, obj: Any):
        cls = obj.__class__
        if cls in self.observed_classes:
            print(f"{cls=} is in {self.observed_classes=}, could have infinite recursion!")
        return Context(
            device=self.device, sampler=self.sampler, config=self.config, observed_classes=self.observed_classes + [cls]
        )


class PriorComponent:
    """
    Base class for components of the prior that tracks info in the context.
    """
    def __init__(self, context: Context):
        self.context = context.get_recursion_context(self)  # track stats

        # for easier access
        self.sampler = self.context.sampler
        self.device = self.context.device
        self.config = copy.copy(context.config)  # copy just to be safe


class RandomTransformer(PriorComponent):
    """
    PriorComponent that acts like a scikit-learn transformer:
    on the first use of __call__, it will do fit_transform(), afterward it will do transform()
    :param context: Context.
    :param allow_override_by_class: Can optionally specify
    that a certain class is allowed to override __call__
    (otherwise it will not be allowed, normally one should implement _fit() and _transform()).
    """
    def __init__(self, context: Context, allow_override_by_class: Optional[type] = None):
        super().__init__(context)
        self.fitted = False
        cls = allow_override_by_class or RandomTransformer
        assert self.__class__.__call__ is cls.__call__

    def _fit(self, *args, **kwargs):
        pass

    def _transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        if not self.fitted:
            self._fit(*args, **kwargs)
            self.fitted = True
        return self._transform(*args, **kwargs)


class RandomTensorTransformer(RandomTransformer):
    """
    RandomTransformer, but typed to accept one torch.Tensor in __call__.
    """
    def __init__(self, context: Context, allow_override_by_class: Optional[type] = None):
        super().__init__(context, allow_override_by_class=allow_override_by_class or RandomTensorTransformer)

    def _fit(self, x: torch.Tensor):
        pass

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor. Should usually be (n_samples, in_features).
        :return: Transformed tensor, typically of shape (n_samples, out_features).
        """
        return super().__call__(x)


class RandomTensorSequential(RandomTensorTransformer):
    """
    Will execute multiple tensor transformers sequentially.
    """
    def __init__(self, context: Context, tfms: List[RandomTensorTransformer]):
        super().__init__(context)
        self.tfms = tfms

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        for tfm in self.tfms:
            x = tfm(x)
        return x


class FeatureSpec:
    """
    Specification for one feature (column) that should be extracted.

    :param group: Group type like "x" or "y" (can be used in the prior to assign different nodes to different groups,
        or to know what the target is for filtering, for example).
    :param cat_size: 0 if numerical, otherwise will be a categorical with (at most) that cardinality.
    """
    def __init__(self, *, group: str = "x", cat_size: int = 0):
        if "_" in group:
            raise ValueError(f"group must not contain an underscore")  # we'll use names of the form {group}_{name}
        self.group = group
        self.cat_size = cat_size


class DatasetProperties:
    """
    A class to specify the desired properties that a dataset should have
    (number of samples, features, feature types, etc.)
    """
    def __init__(
        self,
        n_train: int,
        n_test: int,
        cat_sizes: Optional[Dict[str, List[int]]] = None,
        *,
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
    ):
        """
        :param n_train: Number of train samples.
        :param n_test: Number of test samples.
        :param cat_sizes: (deprecated)
            dict of cat sizes for different parts of the dataset (usually {'x': ..., 'y': ...}).
            Numerical features should have cat size zero.
        :param feature_specs: Feature names and type specifications for each feature.
            Should be used instead of cat_features.
        """
        self.n_train = n_train
        self.n_test = n_test
        if cat_sizes is None:
            assert feature_specs is not None
            self.feature_specs = feature_specs
        else:
            assert feature_specs is None
            self.feature_specs = {
                f"{group}_{i}": FeatureSpec(group=group, cat_size=cat_size)
                for group in cat_sizes
                for i, cat_size in enumerate(cat_sizes[group])
            }


class Dataset:
    """
    Dataset, consisting of tensors, feature specifications, and the graph used to generate them.

    :param tensors: Dictionary of feature names and tensors representing the corresponding column.
    :param feature_specs: Feature names and type specifications for each feature.
    :param kwargs: Other things (like graph, n_train)
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], feature_specs: Dict[str, FeatureSpec], **kwargs):
        assert set(tensors.keys()) == set(feature_specs.keys())
        self.tensors = tensors
        self.feature_specs = feature_specs
        self.kwargs = kwargs

    def get_concat_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Summarizes numerical and categorical features into a single tensor each.
        :return: Dictionary of tensors with names such as "x_num", "x_cat", "x_emb_1", "x_emb_2", "y_num", "y_cat".
        """
        tensor_lists = collections.defaultdict(lambda: [])
        for feature_name, feature_spec in self.feature_specs.items():
            prefix = feature_spec.group
            feat_type = "cat" if feature_spec.cat_size > 1 else "num"
            tensor_lists[f"{prefix}_{feat_type}"].append(self.tensors[feature_name])

        result = dict()

        for name, tensors in tensor_lists.items():
            if "emb" in name:
                for i, tens in enumerate(tensors):
                    result[f"{name}_i"] = tens
            else:
                result[name] = torch.cat(tensors, dim=-1)

        return result
