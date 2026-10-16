# -*- coding: utf-8 -*-
"""
representational_drift/utils/reliability.py

Split-half reliability helpers shared by the gratings and movie analyses.

DMM, October 2026
"""

import numpy as np


def rowwise_corr(a, b):

    a = a - np.mean(a, axis=1, keepdims=True)
    b = b - np.mean(b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a**2, axis=1) * np.sum(b**2, axis=1))

    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.sum(a * b, axis=1) / denom

    r[denom == 0] = np.nan

    return r


def shuffle_pvalue(observed, null):

    n_shuffles = null.shape[1]
    n_ge = np.sum(np.nan_to_num(null, nan=-np.inf) >= observed[:, None], axis=1)
    p = (1 + n_ge) / (1 + n_shuffles)
    p[np.isnan(observed)] = np.nan
    
    return p
