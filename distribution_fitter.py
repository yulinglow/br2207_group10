# -*- coding: utf-8 -*-
"""
BR2207 Distribution Fitting Backend

Calculation functions for the Distribution Fit Explorer app:
    1. Data input and validation
    2. Automated distribution fitting
    3. Goodness-of-fit evaluation
    4. Winning distribution parameters
    5. Moment comparison

This module contains no UI, print or input() code, so it can be imported
safely by the Streamlit app.
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONSTANTS
# ============================================================

MIN_OBSERVATIONS = 30  # n >= 30 for reliable fitting

CONTINUOUS_CANDIDATES = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Uniform": stats.uniform,
    "Beta": stats.beta,
}

DISCRETE_CANDIDATES = {
    "Poisson": stats.poisson,
    "Binomial": stats.binom,
    "Geometric": stats.geom,
    "Negative Binomial": stats.nbinom,
    "Discrete Uniform": stats.randint,
    "Bernoulli": stats.bernoulli,
}

# Readable labels for each distribution's scipy parameters (in scipy's order)
PARAMETER_LABELS = {
    "Normal": ["Mean (loc)", "Standard Deviation (scale)"],
    "Exponential": ["Shift (loc)", "Scale (1/rate)"],
    "Gamma": ["Shape (a)", "Shift (loc)", "Scale"],
    "Lognormal": ["Shape (s)", "Shift (loc)", "Scale"],
    "Uniform": ["Minimum (loc)", "Range (scale)"],
    "Beta": ["Shape 1 (a)", "Shape 2 (b)", "Shift (loc)", "Scale"],
    "Poisson": ["Lambda (mu)", "Shift (loc)"],
    "Binomial": ["Trials (n)", "Probability (p)", "Shift (loc)"],
    "Geometric": ["Probability (p)", "Shift (loc)"],
    "Negative Binomial": ["Number of Successes (n)", "Probability (p)", "Shift (loc)"],
    "Discrete Uniform": ["Minimum Value (low)", "Upper Bound (high, exclusive)", "Shift (loc)"],
    "Bernoulli": ["Probability (p)", "Shift (loc)"],
}


# ============================================================
# 1. DATA INPUT
# ============================================================

def validate_data(values) -> np.ndarray:
    """Convert to a float array and check it is finite and large enough."""
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]

    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains infinite values. Please remove them.")
    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(data)} valid values found. "
            f"At least {MIN_OBSERVATIONS} are needed for reliable fitting."
        )
    return data


def load_from_paste(text: str) -> np.ndarray:
    """Parse numbers pasted as text (comma, semicolon, space or newline separated)."""
    values = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"Could not parse '{token}' as a number.")
    return validate_data(values)


def read_table(source, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file (path or file-like object) into a DataFrame."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(source)
    raise ValueError("Unsupported file type. Please use CSV or Excel.")


def numeric_columns(df: pd.DataFrame) -> list:
    """Return the names of all numeric columns in the DataFrame."""
    columns = list(df.select_dtypes(include=[np.number]).columns)
    if not columns:
        raise ValueError("No numeric column found in the file.")
    return columns


def load_from_file(filepath: str, column: str = None) -> np.ndarray:
    """Load one numeric column (default: the first) from a file path."""
    df = read_table(filepath, filepath)
    if column is None:
        column = numeric_columns(df)[0]
    return validate_data(df[column].to_numpy())


# ============================================================
# 2. AUTOMATED FITTING
# ============================================================

def is_discrete(data: np.ndarray, tol: float = 1e-8) -> bool:
    """Data is treated as discrete if every value is a non-negative integer."""
    return bool(np.all(data >= 0) and np.all(np.abs(data - np.round(data)) < tol))


def _discrete_bounds(dist_name: str, data: np.ndarray) -> dict:
    """Maximum Likelihood search bounds for each discrete candidate (derived from the data)."""
    dmin, dmax = data.min(), data.max()
    prob = (1e-3, 1 - 1e-3)

    bounds = {
        "Poisson": {"mu": (1e-6, dmax * 3 + 10)},
        "Binomial": {"n": (max(dmax, 1), dmax * 5 + 20), "p": prob},
        "Geometric": {"p": prob},
        "Negative Binomial": {"n": (1e-3, dmax * 5 + 20), "p": prob},
        "Discrete Uniform": {"low": (0, dmin + 1), "high": (dmax, dmax * 2 + 10)},
        "Bernoulli": {"p": prob},
    }
    if dist_name not in bounds:
        raise ValueError(f"No bounds defined for '{dist_name}'")
    return bounds[dist_name]


# ============================================================
# 3. GOODNESS OF FIT
# ============================================================

def chisq_gof(fitted_dist, data_int: np.ndarray, n: int, k_params: int):
    """
    Chi-square goodness-of-fit test for a fitted discrete distribution.

    Values are grouped front-to-back so every group has an expected count >= 5
    (Cochran's rule). Returns (chi2_statistic, p_value), or (nan, nan) if there
    are too few groups to run the test.
    """
    lo, hi = data_int.min(), data_int.max()
    support = np.arange(lo, hi + 1)
    observed = pd.Series(data_int).value_counts().reindex(support, fill_value=0).to_numpy()
    expected = fitted_dist.pmf(support) * n

    groups_obs, groups_exp = [], []
    cur_obs, cur_exp = 0, 0.0
    for o, e in zip(observed, expected):
        cur_obs += o
        cur_exp += e
        if cur_exp >= 5:
            groups_obs.append(cur_obs)
            groups_exp.append(cur_exp)
            cur_obs, cur_exp = 0, 0.0

    # Merge any leftover tail into the last group rather than dropping it
    if cur_exp > 0:
        if groups_exp:
            groups_obs[-1] += cur_obs
            groups_exp[-1] += cur_exp
        else:
            groups_obs.append(cur_obs)
            groups_exp.append(cur_exp)

    groups_obs = np.array(groups_obs, dtype=float)
    groups_exp = np.array(groups_exp, dtype=float)
    if len(groups_exp) < 2 or groups_exp.sum() == 0:
        return np.nan, np.nan

    groups_exp *= groups_obs.sum() / groups_exp.sum()
    dof = max(len(groups_obs) - 1 - k_params, 1)

    chi2_stat = np.sum((groups_obs - groups_exp) ** 2 / groups_exp)
    return chi2_stat, stats.chi2.sf(chi2_stat, dof)


def fit_continuous(data: np.ndarray) -> pd.DataFrame:
    """
    Fit each continuous candidate by Maximum Likelihood and evaluate it with:
      - AIC: relative ranking (lower is better)
      - Kolmogorov-Smirnov test: absolute plausibility of each fit on its own
    Returns a DataFrame ranked by AIC (best first).
    """
    min_val = data.min()
    rows = []
    
    for name, dist in CONTINUOUS_CANDIDATES.items():
        try:
            # Skip strictly positive distributions if data contains zero or negative values
            if name in ["Lognormal", "Exponential", "Gamma"] and min_val <= 0:
                raise ValueError(f"Requires strictly positive data (minimum value is {min_val:.2f})")

            # Some scipy fits (e.g. Beta) emit harmless optimiser warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                params = dist.fit(data)

            loglik = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * loglik
            ks_stat, ks_pvalue = stats.kstest(data, dist.name, args=params)

            rows.append({"distribution": name, "params": params, "log_likelihood": loglik,
                         "aic": aic, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue})
        except Exception as e:
            rows.append({"distribution": name, "params": None, "log_likelihood": np.nan,
                         "aic": np.nan, "ks_stat": np.nan,
