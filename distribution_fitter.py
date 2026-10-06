# -*- coding: utf-8 -*-
"""
BR2207 Distribution Fitting Backend

All statistical calculations for the Distribution Fit Explorer live here.
The Streamlit app (app.py) should only handle user input, layout and display.

Key design rule:
    Normal and Lognormal are NOT treated as interchangeable distributions.

    - Normal: support = (-infinity, +infinity)
    - Lognormal: support = (0, +infinity), with loc fixed at 0

Fixing loc=0 for the lognormal is important. If loc is estimated freely,
SciPy can shift a lognormal distribution and make it unnecessarily flexible,
which can cause it to compete with or imitate a roughly symmetric dataset.
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# CONSTANTS
# ============================================================

MIN_OBSERVATIONS = 30

# Continuous candidates.
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

PARAMETER_LABELS = {
    "Normal": [
        "Mean (loc)",
        "Standard Deviation (scale)",
    ],
    "Exponential": [
        "Scale (1/rate)",
    ],
    "Gamma": [
        "Shape (a)",
        "Scale",
    ],
    "Lognormal": [
        "Shape (s)",
        "Scale",
    ],
    "Uniform": [
        "Minimum (loc)",
        "Range (scale)",
    ],
    "Beta": [
        "Shape 1 (a)",
        "Shape 2 (b)",
        "Shift (loc)",
        "Scale",
    ],
    "Poisson": ["Lambda (mu)", "Shift (loc)"],
    "Binomial": [
        "Trials (n)",
        "Probability (p)",
        "Shift (loc)",
    ],
    "Geometric": [
        "Probability (p)",
        "Shift (loc)",
    ],
    "Negative Binomial": [
        "Number of Successes (n)",
        "Probability (p)",
        "Shift (loc)",
    ],
    "Discrete Uniform": [
        "Minimum Value (low)",
        "Upper Bound (high, exclusive)",
        "Shift (loc)",
    ],
    "Bernoulli": [
        "Probability (p)",
        "Shift (loc)",
    ],
}


# ============================================================
# 1. DATA INPUT
# ============================================================

def validate_data(values) -> np.ndarray:
    """Convert values to finite floats and enforce minimum sample size."""
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Data contains infinite values. Please remove them."
        )

    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(data)} valid values found. "
            f"At least {MIN_OBSERVATIONS} are needed for reliable fitting."
        )

    return data


def load_from_paste(text: str) -> np.ndarray:
    """Parse comma, semicolon, space or newline-separated numbers."""
    values = []

    for token in re.split(r"[,;\s]+", text.strip()):
        if token == "":
            continue

        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(
                f"Could not parse '{token}' as a number."
            )

    return validate_data(values)


def read_table(source, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(source)

    if filename.lower().endswith(".csv"):
        return pd.read_csv(source)

    raise ValueError(
        "Unsupported file type. Please use CSV or Excel."
    )


def numeric_columns(df: pd.DataFrame) -> list:
    """Return numeric columns."""
    columns = list(
        df.select_dtypes(include=[np.number]).columns
    )

    if not columns:
        raise ValueError("No numeric column found in the file.")

    return columns


def load_from_file(filepath: str, column: str = None) -> np.ndarray:
    """Load one numeric column from a file."""
    df = read_table(filepath, filepath)

    if column is None:
        column = numeric_columns(df)[0]

    return validate_data(df[column].to_numpy())


# ============================================================
# 2. DATA TYPE DETECTION
# ============================================================

def is_discrete(data: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Treat data as discrete only when every observation is a
    non-negative integer.
    """
    return bool(
        np.all(data >= 0)
        and np.all(np.abs(data - np.round(data)) < tol)
    )


# ============================================================
# 3. DISTRIBUTION-SPECIFIC FITTING
# ============================================================

def _fit_continuous_distribution(name, dist, data):
    """
    Fit one continuous distribution.

    Important:
    - Normal estimates loc and scale.
    - Lognormal is fitted with loc fixed at 0.
    - Exponential is fitted with loc fixed at 0.
    - Gamma is fitted with loc fixed at 0.
    - Beta remains a four-parameter bounded distribution because its
      location and scale define the observed interval.
    """
    if name == "Lognormal":
        if np.any(data <= 0):
            raise ValueError(
                "Lognormal requires all observations to be strictly positive."
            )

        return dist.fit(data, floc=0)

    if name == "Exponential":
        if np.any(data < 0):
            raise ValueError(
                "Exponential requires non-negative observations."
            )

        return dist.fit(data, floc=0)

    if name == "Gamma":
        if np.any(data <= 0):
            raise ValueError(
                "Gamma requires all observations to be strictly positive."
            )

        return dist.fit(data, floc=0)

    return dist.fit(data)


# ============================================================
# 4. DISCRETE FITTING HELPERS
# ============================================================

def _discrete_bounds(dist_name: str, data: np.ndarray) -> dict:
    """Maximum-likelihood search bounds derived from the data."""
    dmin, dmax = data.min(), data.max()
    prob = (1e-3, 1 - 1e-3)

    bounds = {
        "Poisson": {
            "mu": (1e-6, dmax * 3 + 10),
        },
        "Binomial": {
            "n": (max(dmax, 1), dmax * 5 + 20),
            "p": prob,
        },
        "Geometric": {
            "p": prob,
        },
        "Negative Binomial": {
            "n": (1e-3, dmax * 5 + 20),
            "p": prob,
        },
        "Discrete Uniform": {
            "low": (0, dmin + 1),
            "high": (dmax, dmax * 2 + 10),
        },
        "Bernoulli": {
            "p": prob,
        },
    }

    if dist_name not in bounds:
        raise ValueError(
            f"No bounds defined for '{dist_name}'"
        )

    return bounds[dist_name]


# ============================================================
# 5. GOODNESS OF FIT
# ============================================================

def chisq_gof(fitted_dist, data_int, n, k_params):
    """
    Chi-square goodness-of-fit test for a fitted discrete distribution.

    Adjacent values are grouped until each expected frequency is at
    least 5, following the usual Cochran-style rule.
    """
    lo, hi = data_int.min(), data_int.max()

    support = np.arange(lo, hi + 1)

    observed = (
        pd.Series(data_int)
        .value_counts()
        .reindex(support, fill_value=0)
        .to_numpy()
    )

    expected = fitted_dist.pmf(support) * n

    groups_obs = []
    groups_exp = []
    current_obs = 0
    current_exp = 0.0

    for observed_count, expected_count in zip(
        observed,
        expected,
    ):
        current_obs += observed_count
        current_exp += expected_count

        if current_exp >= 5:
            groups_obs.append(current_obs)
            groups_exp.append(current_exp)
            current_obs = 0
            current_exp = 0.0

    if current_exp > 0:
        if groups_exp:
            groups_obs[-1] += current_obs
            groups_exp[-1] += current_exp
        else:
            groups_obs.append(current_obs)
            groups_exp.append(current_exp)

    groups_obs = np.asarray(groups_obs, dtype=float)
    groups_exp = np.asarray(groups_exp, dtype=float)

    if len(groups_exp) < 2 or groups_exp.sum() == 0:
        return np.nan, np.nan

    groups_exp *= groups_obs.sum() / groups_exp.sum()

    dof = max(
        len(groups_obs) - 1 - k_params,
        1,
    )

    chi2_stat = np.sum(
        (groups_obs - groups_exp) ** 2 / groups_exp
    )

    return chi2_stat, stats.chi2.sf(chi2_stat, dof)


# ============================================================
# 6. CONTINUOUS FITTING
# ============================================================

def fit_continuous(data: np.ndarray) -> pd.DataFrame:
    """
    Fit all continuous candidates.

    Ranking:
        1. Maximum likelihood estimation
        2. AIC, lower is better

    The KS statistic/p-value is also reported as an absolute
    goodness-of-fit diagnostic.

    Lognormal is intentionally constrained to loc=0 so that it retains
    its standard positive-only support.
    """
    rows = []

    for name, dist in CONTINUOUS_CANDIDATES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                params = _fit_continuous_distribution(
                    name,
                    dist,
                    data,
                )

            logpdf_values = dist.logpdf(
                data,
                *params,
            )

            if not np.all(np.isfinite(logpdf_values)):
                raise ValueError(
                    "Data contains observations outside the "
                    f"{name} distribution's support."
                )

            loglik = np.sum(logpdf_values)

            # Number of free parameters actually estimated.
            # Lognormal, Gamma and Exponential have loc fixed at 0.
            if name in {"Lognormal", "Gamma"}:
                k = 2
            elif name == "Exponential":
                k = 1
            else:
                k = len(params)

            aic = 2 * k - 2 * loglik

            ks_stat, ks_pvalue = stats.kstest(
                data,
                dist.name,
                args=params,
            )

            rows.append(
                {
                    "distribution": name,
                    "params": params,
                    "log_likelihood": loglik,
                    "num_parameters": k,
                    "aic": aic,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "distribution": name,
                    "params": None,
                    "log_likelihood": np.nan,
                    "num_parameters": np.nan,
                    "aic": np.nan,
                    "ks_stat": np.nan,
                    "ks_pvalue": np.nan,
                    "error": str(e),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("aic", na_position="last")
        .reset_index(drop=True)
    )


# ============================================================
# 7. DISCRETE FITTING
# ============================================================

def fit_discrete(data: np.ndarray) -> pd.DataFrame:
    """
    Fit all discrete candidates using maximum likelihood and AIC.
    """
    n = len(data)
    data_int = data.astype(int)
    rows = []

    for name, dist in DISCRETE_CANDIDATES.items():
        try:
            bounds = _discrete_bounds(name, data)

            fitted = stats.fit(
                dist,
                data,
                bounds=bounds,
            )

            params = tuple(fitted.params)
            fitted_dist = dist(*params)

            loglik = np.sum(
                fitted_dist.logpmf(data)
            )

            if not np.isfinite(loglik):
                raise ValueError(
                    f"Data falls outside {name}'s support."
                )

            k = len(bounds)

            aic = 2 * k - 2 * loglik

            chi2_stat, chi2_pvalue = chisq_gof(
                fitted_dist,
                data_int,
                n,
                k,
            )

            rows.append(
                {
                    "distribution": name,
                    "params": params,
                    "log_likelihood": loglik,
                    "num_parameters": k,
                    "aic": aic,
                    "chi2_stat": chi2_stat,
                    "chi2_pvalue": chi2_pvalue,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "distribution": name,
                    "params": None,
                    "log_likelihood": np.nan,
                    "num_parameters": np.nan,
                    "aic": np.nan,
                    "chi2_stat": np.nan,
                    "chi2_pvalue": np.nan,
                    "error": str(e),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("aic", na_position="last")
        .reset_index(drop=True)
    )


# ============================================================
# 8. MAIN FITTING ENTRY POINT
# ============================================================

def fit_best_distribution(data: np.ndarray):
    """
    Single entry point used by app.py.

    Detects discrete/continuous data and runs the appropriate
    distribution-fitting pipeline.
    """
    discrete = is_discrete(data)

    if discrete:
        results = fit_discrete(data)
    else:
        results = fit_continuous(data)

    return results, discrete


# ============================================================
# 9. WINNING DISTRIBUTION
# ============================================================

def _get_winner(results_df: pd.DataFrame):
    """Return the best valid row."""
    valid = results_df.dropna(
        subset=["aic"]
    )

    if valid.empty:
        raise ValueError(
            "No valid winning distribution found."
        )

    return valid.iloc[0]


def get_fitted_distribution(
    results_df: pd.DataFrame,
    discrete_flag: bool,
):
    """Return (name, frozen scipy distribution)."""
    best = _get_winner(results_df)

    candidates = (
        DISCRETE_CANDIDATES
        if discrete_flag
        else CONTINUOUS_CANDIDATES
    )

    return (
        best["distribution"],
        candidates[best["distribution"]](
            *best["params"]
        ),
    )


def extract_winning_parameters(
    results_df: pd.DataFrame,
    discrete_flag: bool,
):
    """Return winning distribution and readable parameters."""
    best = _get_winner(results_df)

    name = best["distribution"]
    params = best["params"]

    labels = PARAMETER_LABELS.get(
        name,
        [f"Param {i + 1}" for i in range(len(params))],
    )

    formatted = dict(
        zip(labels, params)
    )

    if name == "Uniform":
        formatted = {
            "Minimum (loc)": params[0],
            "Maximum": params[0] + params[1],
            "Range (scale)": params[1],
        }

    return name, formatted


# ============================================================
# 10. MOMENTS
# ============================================================

def _ordinal(n: int) -> str:
    return {
        1: "1st",
        2: "2nd",
        3: "3rd",
    }.get(n, f"{n}th")


def calculate_moments(
    data: np.ndarray,
    results_df: pd.DataFrame,
    discrete_flag: bool,
):
    """
    Compare observed moments with moments of the winning distribution.
    """
    dist_name, fitted_dist = get_fitted_distribution(
        results_df,
        discrete_flag,
    )

    raw_rows = []

    for order in range(1, 5):
        empirical = np.mean(data ** order)
        theoretical = fitted_dist.moment(order)

        raw_rows.append(
            {
                "Moment": f"{_ordinal(order)} Raw Moment",
                "Raw Data": empirical,
                "Winning Distribution": theoretical,
                "Absolute Difference": abs(
                    empirical - theoretical
                ),
            }
        )

    raw_moments_df = pd.DataFrame(raw_rows)

    fitted_mean, fitted_var, fitted_skew, fitted_excess_kurt = (
        fitted_dist.stats(moments="mvsk")
    )

    characteristics_df = pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Variance",
                "Skewness",
                "Kurtosis",
            ],
            "Raw Data": [
                np.mean(data),
                np.var(data, ddof=0),
                stats.skew(data, bias=True),
                stats.kurtosis(
                    data,
                    fisher=False,
                    bias=True,
                ),
            ],
            "Winning Distribution": [
                float(fitted_mean),
                float(fitted_var),
                float(fitted_skew),
                float(fitted_excess_kurt) + 3,
            ],
        }
    )

    characteristics_df["Absolute Difference"] = abs(
        characteristics_df["Raw Data"]
        - characteristics_df["Winning Distribution"]
    )

    return (
        dist_name,
        raw_moments_df,
        characteristics_df,
    )
