# Copied from https://github.com/PriorLabs/tabpfn-time-series
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    Subclasses must implement ``generate`` to add feature columns to a
    DataFrame. Instances are callable via ``__call__``, which delegates
    to ``generate``.
    """

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to augment with features.

        Returns
        -------
        pd.DataFrame
            DataFrame with added feature columns.
        """
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.__dict__}"

    def __repr__(self) -> str:
        return self.__str__()
