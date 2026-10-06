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
            # Use explicit dist.cdf to avoid positional argument mapping issues with internal string lookups
            ks_stat, ks_pvalue = stats.kstest(data, dist.cdf, args=params)

            rows.append({"distribution": name, "params": params, "log_likelihood": loglik,
                         "aic": aic, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue})
        except Exception as e:
            rows.append({"distribution": name, "params": None, "log_likelihood": np.nan,
                         "aic": np.nan, "ks_stat": np.nan, "ks_pvalue": np.nan,
                         "error": str(e)})

    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def fit_discrete(data: np.ndarray) -> pd.DataFrame:
    """
    Fit each discrete candidate by Maximum Likelihood and evaluate it with:
      - AIC: relative ranking (lower is better)
      - Chi-square test: absolute plausibility of each fit on its own
    Returns a DataFrame ranked by AIC (best first).
    """
    n = len(data)
    data_int = data.astype(int)
    rows = []

    for name, dist in DISCRETE_CANDIDATES.items():
        try:
            bounds = _discrete_bounds(name, data)
            params = tuple(stats.fit(dist, data, bounds=bounds).params)
            fitted_dist = dist(*params)

            loglik = np.sum(fitted_dist.logpmf(data))
            if not np.isfinite(loglik):
                raise ValueError(f"Data falls outside {name}'s support - not a viable candidate")

            k = len(bounds)  # number of estimated (shape) parameters; loc is fixed at 0
            aic = 2 * k - 2 * loglik
            chi2_stat, chi2_pvalue = chisq_gof(fitted_dist, data_int, n, k)

            rows.append({"distribution": name, "params": params, "log_likelihood": loglik,
                         "aic": aic, "chi2_stat": chi2_stat, "chi2_pvalue": chi2_pvalue})
        except Exception as e:
            rows.append({"distribution": name, "params": None, "log_likelihood": np.nan,
                         "aic": np.nan, "chi2_stat": np.nan, "chi2_pvalue": np.nan,
                         "error": str(e)})

    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def fit_best_distribution(data: np.ndarray):
    """
    Single entry point for fitting (the app must call this).

    Detects whether the data is discrete or continuous, runs the matching
    pipeline, and returns (ranked_results_df, discrete_flag).
    """
    discrete = is_discrete(data)
    results = fit_discrete(data) if discrete else fit_continuous(data)
    return results, discrete


# ============================================================
# 4. PARAMETERS
# ============================================================

def _get_winner(results_df: pd.DataFrame):
    """Return the top-ranked row, or raise if no distribution was fitted successfully."""
    if results_df.empty or results_df.iloc[0]["params"] is None:
        raise ValueError("No valid winning distribution found.")
    return results_df.iloc[0]


def get_fitted_distribution(results_df: pd.DataFrame, discrete_flag: bool):
    """Return (name, frozen scipy distribution) for the winning fit."""
    best = _get_winner(results_df)
    candidates = DISCRETE_CANDIDATES if discrete_flag else CONTINUOUS_CANDIDATES
    return best["distribution"], candidates[best["distribution"]](*best["params"])


def extract_winning_parameters(results_df: pd.DataFrame, discrete_flag: bool):
    """Return (distribution_name, {readable label: value}) for the winning fit."""
    best = _get_winner(results_df)
    name, params = best["distribution"], best["params"]

    labels = PARAMETER_LABELS.get(name, [f"Param {i + 1}" for i in range(len(params))])
    formatted = dict(zip(labels, params))

    if name == "Uniform":  # also report the upper end of the range
        formatted = {
            "Minimum (loc)": params[0],
            "Maximum": params[0] + params[1],
            "Range (scale)": params[1],
        }
    return name, formatted


# ============================================================
# 5. MOMENTS
# ============================================================

def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def calculate_moments(data: np.ndarray, results_df: pd.DataFrame, discrete_flag: bool):
    """
    Compare the raw data with the winning distribution.

    Returns (distribution_name, raw_moments_df, characteristics_df):
      - raw_moments_df: first four raw moments
      - characteristics_df: mean, variance, skewness, kurtosis (Pearson)
    """
    dist_name, fitted_dist = get_fitted_distribution(results_df, discrete_flag)

    # Part A: first four raw moments, E[X^r]
    raw_rows = []
    for order in range(1, 5):
        empirical = np.mean(data ** order)
        theoretical = fitted_dist.moment(order)
        raw_rows.append({
            "Moment": f"{_ordinal(order)} Raw Moment",
            "Raw Data": empirical,
            "Winning Distribution": theoretical,
            "Absolute Difference": abs(empirical - theoretical),
        })
    raw_moments_df = pd.DataFrame(raw_rows)

    # Part B: distribution characteristics
    fitted_mean, fitted_var, fitted_skew, fitted_excess_kurt = fitted_dist.stats(moments="mvsk")

    characteristics_df = pd.DataFrame({
        "Statistic": ["Mean", "Variance", "Skewness", "Kurtosis"],
        "Raw Data": [
            np.mean(data),
            np.var(data, ddof=0),
            stats.skew(data, bias=True),
            stats.kurtosis(data, fisher=False, bias=True),
        ],
        "Winning Distribution": [
            float(fitted_mean),
            float(fitted_var),
            float(fitted_skew),
            float(fitted_excess_kurt) + 3,  # scipy gives excess kurtosis; add 3 for Pearson
        ],
    })
    characteristics_df["Absolute Difference"] = abs(
        characteristics_df["Raw Data"] - characteristics_df["Winning Distribution"]
    )

    return dist_name, raw_moments_df, characteristics_df
