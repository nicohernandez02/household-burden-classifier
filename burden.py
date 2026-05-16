import numpy as np
import pandas as pd


def my474_predict(X, fitted_object):
    """
    Return predicted probabilities of class 1 for each row of X.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor columns only (no 'y' column), arbitrary number of rows.
    fitted_object : sklearn Pipeline
        The unpickled fitted pipeline (LR + target encoding + feature engineering).

    Returns
    -------
    np.ndarray
        1-D array of floats in [0, 1], one value per row of X.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        probs = fitted_object.predict_proba(X)[:, 1]
    return np.array(probs, dtype=float)
