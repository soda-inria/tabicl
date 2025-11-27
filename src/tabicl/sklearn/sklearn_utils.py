"""Utility functions for sklearn compatibility.

This module provides some scikit-learn utils that can be used across different
versions of scikit-learn, eliminating the need for version-specific handling.

Copied from: https://github.com/scikit-learn/scikit-learn/blob/1eb422d6c5/sklearn/utils/validation.py
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np


def _is_pandas_df(X) -> bool:
    """Return True if the X is a pandas dataframe."""
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(X, pd.DataFrame)


def _get_feature_names(X):
    """Get feature names from X.

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    if _is_pandas_df(X):
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return None

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # mixed type of string and non-string is not supported
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    # Only feature names of all strings are supported
    if len(types) == 1 and types[0] == "str":
        return feature_names

    return None


def _check_feature_names(estimator, X, *, reset: bool) -> None:
    """Set or check the `feature_names_in_` attribute of an estimator.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to validate the input for.

    X : {ndarray, dataframe} of shape (n_samples, n_features)
        The input samples.

    reset : bool
        Whether to reset the `feature_names_in_` attribute.
        If False, the input will be checked for consistency with
        feature names of data provided when reset was last True.
    """
    if reset:
        feature_names_in = _get_feature_names(X)
        if feature_names_in is not None:
            estimator.feature_names_in_ = feature_names_in
        elif hasattr(estimator, "feature_names_in_"):
            # Delete the attribute when the estimator is fitted on a new dataset
            # that has no feature names.
            delattr(estimator, "feature_names_in_")
        return

    fitted_feature_names = getattr(estimator, "feature_names_in_", None)
    X_feature_names = _get_feature_names(X)

    if fitted_feature_names is None and X_feature_names is None:
        return

    if X_feature_names is not None and fitted_feature_names is None:
        warnings.warn(f"X has feature names, but {estimator.__class__.__name__} was fitted without feature names")
        return

    if X_feature_names is None and fitted_feature_names is not None:
        warnings.warn(
            f"X does not have valid feature names, but {estimator.__class__.__name__} was fitted with feature names"
        )
        return

    # validate the feature names against the `feature_names_in_` attribute
    if len(fitted_feature_names) != len(X_feature_names) or np.any(fitted_feature_names != X_feature_names):
        message = "The feature names should match those that were passed during fit.\n"
        fitted_feature_names_set = set(fitted_feature_names)
        X_feature_names_set = set(X_feature_names)

        unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
        missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

        def add_names(names):
            output = ""
            max_n_names = 5
            for i, name in enumerate(names):
                if i >= max_n_names:
                    output += "- ...\n"
                    break
                output += f"- {name}\n"
            return output

        if unexpected_names:
            message += "Feature names unseen at fit time:\n"
            message += add_names(unexpected_names)

        if missing_names:
            message += "Feature names seen at fit time, yet now missing:\n"
            message += add_names(missing_names)

        if not missing_names and not unexpected_names:
            message += "Feature names must be in the same order as they were in fit.\n"

        raise ValueError(message)


def _use_interchange_protocol(X) -> bool:
    """Use interchange protocol for non-pandas dataframes that follow the protocol."""
    return not _is_pandas_df(X) and hasattr(X, "__dataframe__")


def _num_features(X) -> int:
    """Return the number of features in an array-like X.

    Parameters
    ----------
    X : array-like
        array-like to get the number of features.

    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of features from X of type {type_name}"

    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err


def _check_n_features(estimator, X, reset: bool) -> None:
    """Set the `n_features_in_` attribute, or check against it on an estimator.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to validate the input for.

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    reset : bool
        If True, the `n_features_in_` attribute is set to `X.shape[1]`.
        If False and the attribute exists, then check that it is equal to
        `X.shape[1]`. If False and the attribute does *not* exist, then
        the check is skipped.
    """
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(estimator, "n_features_in_"):
            raise ValueError(
                "X does not contain any features, but "
                f"{estimator.__class__.__name__} is expecting "
                f"{estimator.n_features_in_} features"
            ) from e
        return

    if reset:
        estimator.n_features_in_ = n_features
        return

    if not hasattr(estimator, "n_features_in_"):
        return

    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )


def _num_samples(x) -> int:
    """Return number of samples in array-like x."""
    import numbers

    message = "Expected sequence or array-like, got %s" % type(x)

    if hasattr(x, "fit") and callable(x.fit):
        raise TypeError(message)

    if _use_interchange_protocol(x):
        return x.__dataframe__().num_rows()

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Input should have at least 1 dimension i.e. satisfy "
                f"`len(x.shape) > 0`, got scalar `{x!r}` instead."
            )
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_consistent_length(*arrays) -> None:
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [_num_samples(X) for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % [int(length) for length in lengths]
        )


def _check_y(y, multi_output: bool = False, y_numeric: bool = False, estimator=None):
    """Isolated part of check_X_y dedicated to y validation."""
    from sklearn.utils.validation import check_array, column_or_1d

    if multi_output:
        y = check_array(
            y,
            accept_sparse="csr",
            ensure_all_finite=True,
            ensure_2d=False,
            dtype=None,
            input_name="y",
            estimator=estimator,
        )
    else:
        y = column_or_1d(y, warn=True)

    if y_numeric and hasattr(y.dtype, "kind") and y.dtype.kind == "O":
        y = y.astype(np.float64)

    return y


def validate_data(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    reset: bool = True,
    validate_separately: bool | tuple = False,
    skip_check_array: bool = False,
    **check_params: Any,
):
    """Validate input data and set or check feature names and counts of the input.

    This is a standalone version of sklearn's validate_data function that works
    across different sklearn versions.

    Parameters
    ----------
    _estimator : estimator instance
        The estimator to validate the input for.

    X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features), \
            default='no validation'
        The input samples.
        If `'no_validation'`, no validation is performed on `X`.

    y : array-like of shape (n_samples,), default='no_validation'
        The targets.
        - If `None`, only X is validated.
        - If `'no_validation'`, only X is validated.

    reset : bool, default=True
        Whether to reset the `n_features_in_` attribute.
        If False, the input will be checked for consistency with data
        provided when reset was last True.

    validate_separately : False or tuple of dicts, default=False
        Only used if `y` is not `None`.
        If `False`, call check_X_y. Else, it must be a tuple of kwargs
        to be used for calling check_array on X and y respectively.

    skip_check_array : bool, default=False
        If `True`, `X` and `y` are unchanged and only `feature_names_in_` and
        `n_features_in_` are checked.

    **check_params : kwargs
        Parameters passed to check_array or check_X_y.

    Returns
    -------
    out : {ndarray, sparse matrix} or tuple of these
        The validated input. A tuple is returned if both `X` and `y` are validated.
    """
    from sklearn.utils.validation import check_array, check_X_y

    _check_feature_names(_estimator, X, reset=reset)

    no_val_X = isinstance(X, str) and X == "no_validation"
    no_val_y = y is None or (isinstance(y, str) and y == "no_validation")

    if no_val_X and no_val_y:
        raise ValueError("Validation should be done on X, y or both.")

    default_check_params = {"estimator": _estimator}
    check_params = {**default_check_params, **check_params}

    if skip_check_array:
        if not no_val_X and no_val_y:
            out = X
        elif no_val_X and not no_val_y:
            out = y
        else:
            out = X, y
    elif not no_val_X and no_val_y:
        out = check_array(X, input_name="X", **check_params)
    elif no_val_X and not no_val_y:
        out = _check_y(y, **check_params)
    else:
        if validate_separately:
            check_X_params, check_y_params = validate_separately
            if "estimator" not in check_X_params:
                check_X_params = {**default_check_params, **check_X_params}
            X = check_array(X, input_name="X", **check_X_params)
            if "estimator" not in check_y_params:
                check_y_params = {**default_check_params, **check_y_params}
            y = check_array(y, input_name="y", **check_y_params)
        else:
            X, y = check_X_y(X, y, **check_params)
        out = X, y

    if not no_val_X and check_params.get("ensure_2d", True):
        _check_n_features(_estimator, X, reset=reset)

    return out
