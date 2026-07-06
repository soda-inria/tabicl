from typing import List
import torch

from tabicl.prior.graph_lib._base import RandomTransformer, Context
from tabicl.prior.graph_lib._function import RandomFunction


class RandomMultiFunction(RandomTransformer):
    """
    Random function that can handle multiple input nodes.
    """
    def __init__(self, context: Context, out_features: int):
        super().__init__(context)
        self.out_features = out_features

    def _fit(self, x: List[torch.Tensor]):
        if self.config.multi_fct_types == 'default':
            func_type = self.sampler.choice(
                "multi_function_func_type",
                [
                    RandomConcatMultiFunction,
                    RandomAggregationMultiFunction,
                ],
            )
        elif self.config.multi_fct_types == 'concat':
            func_type = RandomConcatMultiFunction
        elif self.config.multi_fct_types == 'agg':
            func_type = RandomAggregationMultiFunction
        else:
            raise ValueError(f'Unknown option {self.config.multi_function_types=}')
        self.func_ = func_type(self.context, self.out_features)

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.func_(x)


class RandomConcatMultiFunction(RandomMultiFunction):
    def _fit(self, x: List[torch.Tensor]):
        in_features = sum([z.shape[-1] for z in x])
        self.func_ = RandomFunction(self.context, in_features, self.out_features)

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.func_(torch.cat(x, dim=-1))


class RandomAggregationMultiFunction(RandomMultiFunction):
    def _fit(self, x: List[torch.Tensor]):
        self.op_type_ = self.sampler.choice("aggregation_op_type", ["sum", "product", "max", "logsumexp"])
        self.funcs_ = [RandomFunction(self.context, tens.shape[-1], self.out_features) for tens in x]

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        out_cat = torch.stack([f(tens) for f, tens in zip(self.funcs_, x)], dim=0)
        if self.op_type_ == "sum":
            return torch.sum(out_cat, dim=0)
        elif self.op_type_ == "product":
            return torch.prod(out_cat, dim=0)
        elif self.op_type_ == "max":
            return torch.max(out_cat, dim=0).values
        elif self.op_type_ == "logsumexp":
            return torch.logsumexp(out_cat, dim=0)
        else:
            raise ValueError(f'Unknown op_type "{self.op_type_}"')
