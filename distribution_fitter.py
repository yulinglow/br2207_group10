# -*- coding: utf-8 -*-
"""
BR2207 Distribution Fitting Backend

Cleaned backend module combining the group's calculation functions:
1. Data input
2. Automated distribution fitting
3. Goodness-of-fit evaluation
4. Winning distribution parameters
5. Moment comparison

Testing, Monte Carlo validation, terminal input(), and print-based UI code
have been removed so this module can be safely imported by Streamlit.
"""

# **CHLOE'S PORTION (TASKS 1 - 3)**

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

## **1. DATA INPUT**

# create function to load numeric data from Excel file

def load_from_file(filepath: str, column: str = None) -> np.ndarray:
    if filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    if column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric column found in the file.")
        column = numeric_cols[0]

    data = df[column].dropna().to_numpy(dtype=float)

    if len(data) < 30:          # n >= 30 for reliable results
        raise ValueError(f"Only {len(data)} valid values found — need at least 30 for reliable fitting.")
    return data

# create function to parse numeric data pasted as text (comma, space, or newline separated)

def load_from_paste(text: str) -> np.ndarray:
    import re
    tokens = re.split(r"[,\s]+", text.strip())
    values = []
    for t in tokens:
        if t == "":
            continue
        try:
            values.append(float(t))
        except ValueError:
            raise ValueError(f"Could not parse '{t}' as a number.")

    data = np.array(values, dtype=float)

    if len(data) < 30:         # n >= 30 for reliable results
        raise ValueError(f"Only {len(data)} valid values found — need at least 30 for reliable fitting.")
    return data


#### test that data input functions work (delete below later)

# test pasted text function
#pasted_test = load_from_paste("12.1, 14.3 15.0\n9.8, 13.2, 16.7, 11.4, 10.9, 14.8")
#print("Parsed from paste:", pasted_test)

# test reading csv file path function
#pd.DataFrame({"value": np.random.normal(50, 5, 30)}).to_csv("test_data.csv", index=False)
#csv_test = load_from_file("test_data.csv")
#print("Parsed from CSV, first 5:", csv_test[:5], "... total:", len(csv_test))

## **2. AUTOMATED FITTING**

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

# data is treated as discrete/count data if every value is a non-negative integer

def is_discrete(data: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.all(data >= 0) and np.all(np.abs(data - np.round(data)) < tol))


# search bounds for Maxmimum Likelihood Estimation (MLE) fitting of each discrete candidate
# derived from the data itself rather than hardcoded

def _discrete_bounds(dist_name: str, data: np.ndarray) -> dict:
    dmin, dmax = data.min(), data.max()
    if dist_name == "Poisson":
        return {"mu": (1e-6, dmax * 3 + 10)}
    elif dist_name == "Binomial":
        return {"n": (max(dmax, 1), dmax * 5 + 20), "p": (1e-3, 1 - 1e-3)}
    elif dist_name == "Geometric":
        return {"p": (1e-3, 1 - 1e-3)}
    elif dist_name == "Negative Binomial":
        return {"n": (1e-3, dmax * 5 + 20), "p": (1e-3, 1 - 1e-3)}
    elif dist_name == "Discrete Uniform":
        return {"low": (0, dmin + 1), "high": (dmax, dmax * 2 + 10)}
    elif dist_name == "Bernoulli":
        return {"p": (1e-3, 1 - 1e-3)}
    else:
        raise ValueError(f"No bounds defined for '{dist_name}'")

#### Create definition chisq_gof for Chi-square Goodness-of-Fit test.
#Groups the distribution's support into bins front-to-back so every bin has an expected count >= 5 (Cochran's rule — the standard requirement for this test to be statistically valid), rather than assuming bins line up neatly, which breaks for distributions like Geometric whose support doesn't start at 0.

def chisq_gof(fitted_dist, data_int: np.ndarray, n: int, k_params: int):
    lo, hi = data_int.min(), data_int.max()              # smallest and largest values seen in the data
    support = np.arange(lo, hi + 1)                      # every whole number between lo and hi, e.g. [0,1,2,...,16]
    observed_counts = pd.Series(data_int).value_counts().reindex(support, fill_value=0).to_numpy()
    # ^ how many times each value in `support` actually appeared in the data
    expected_counts = np.array([fitted_dist.pmf(k) for k in support]) * n
    # ^ how many times the fitted distribution predicts each value should appear

    groups_obs, groups_exp = [], []                       # group final totals for observed/expected counts
    cur_obs, cur_exp = 0, 0.0
    for o, e in zip(observed_counts, expected_counts):
        cur_obs += o
        cur_exp += e
        if cur_exp >= 5:                                  # Cochran's rule
            groups_obs.append(cur_obs)
            groups_exp.append(cur_exp)
            cur_obs, cur_exp = 0, 0.0
    if cur_exp > 0:                                       # leftover tail — merge into the last group rather than drop it
        if groups_exp:
            groups_obs[-1] += cur_obs
            groups_exp[-1] += cur_exp
        else:
            groups_obs.append(cur_obs)
            groups_exp.append(cur_exp)

    groups_obs, groups_exp = np.array(groups_obs, dtype=float), np.array(groups_exp, dtype=float) # convert lists into numpy arrays
    if len(groups_exp) < 2 or groups_exp.sum() == 0:
        return np.nan, np.nan                             # not enough distinct bins to run the test meaningfully

    groups_exp = groups_exp * (groups_obs.sum() / groups_exp.sum())
    dof = max(len(groups_obs) - 1 - k_params, 1)          # degrees of freedom: adjusts the test for how many parameters were estimated when fitting the distribution

    chi2_stat = np.sum((groups_obs - groups_exp) ** 2 / groups_exp)
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)          # converts chi-square statistic into a p-value (0-1 scale)
    return chi2_stat, p_value

## **3. GOODNESS OF FIT**

#### Create definition fit_continuous to fit each continuous candidate distribution to the data using Maximum Likelihood Estimation (MLE), then evaluate the quality of each fit using 2 different metrics:
#Akaike Information Criterion (AIC) - relative ranking:
#- Identifies which candidate is the best fit relative to the others tested.

#Kolmogorov-Smirnov (KS) test - absolute plausibility:
#- tests whether the specific distribution, on its own, is a statistically believable fit for the data (independent of other candidates' performance).


def fit_continuous(data: np.ndarray) -> pd.DataFrame:
    import warnings

    rows = []

    for name, dist in CONTINUOUS_CANDIDATES.items():
        try:

            # Suppress numerical optimisation warnings produced
            # internally by some scipy distributions such as Beta
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                params = dist.fit(data)

            loglik = np.sum(dist.logpdf(data, *params))

            k = len(params)
            aic = 2 * k - 2 * loglik

            ks_stat, ks_pvalue = stats.kstest(
                data,
                dist.name,
                args=params
            )

            rows.append({
                "distribution": name,
                "params": params,
                "log_likelihood": loglik,
                "aic": aic,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue
            })

        except Exception as e:

            rows.append({
                "distribution": name,
                "params": None,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "error": str(e)
            })

    return pd.DataFrame(rows).sort_values(
        "aic"
    ).reset_index(drop=True)

#### Create definition fit_discrete to fit each discrete candidate distribution to the data using Maximum Likelihood Estimation (MLE), then evaluate the quality of each fit using 2 different metrics:
#Akaike Information Criterion (AIC) - relative ranking:
#- Identifies which candidate is the best fit relative to the others tested.

#Chi-square Goodness-of-Fit test - absolute plausibility:
#- tests whether the specific distribution, on its own, is a statistically believable fit for the data (independent of other candidates' performance).

def fit_discrete(data: np.ndarray) -> pd.DataFrame:
    n = len(data)
    data_int = data.astype(int)
    rows = []

    for name, dist in DISCRETE_CANDIDATES.items():             # iterate through each candidate distribution
        try:
            bounds = _discrete_bounds(name, data)
            fit_result = stats.fit(dist, data, bounds=bounds)
            params = tuple(fit_result.params)                  # the best-fitting parameter values found
            fitted_dist = dist(*params)
            loglik = np.sum(fitted_dist.logpmf(data))          # log-likelihood: how probable the data is under this fit
            if not np.isfinite(loglik):
            # e.g. data has a 0 but Bernoulli/Geometric's support doesn't include it at these params
                raise ValueError(f"data falls outside {name}'s support given fitted params — not a viable candidate")

            k = len(bounds)
            aic = 2 * k - 2 * loglik                           # AIC criterion: lower = better fit, penalized for extra parameters
            chi2_stat, chi2_pvalue = chisq_gof(fitted_dist, data_int, n, k)       # chi-square GOF statistic and p-value

            rows.append({"distribution": name, "params": params, "log_likelihood": loglik,
                         "aic": aic, "chi2_stat": chi2_stat, "chi2_pvalue": chi2_pvalue})
        except Exception as e:
            rows.append({"distribution": name, "params": None, "log_likelihood": np.nan,
                         "aic": np.nan, "chi2_stat": np.nan, "chi2_pvalue": np.nan, "error": str(e)})

    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)  # sort best (lowest AIC) fit first

#### Create definition fit_best_distribution to act as the single entry point for the whole fitting process
#- Detects whether the dataset is discrete (whole numbers, e.g. counts) or continuous
 # (decimal values), using is_discrete.
#- Runs the matching pipeline — fit_discrete or fit_continuous — depending on that result.
#- Returns a ranked table of every candidate distribution tried (best fit first), along with
 # a flag indicating whether the discrete or continuous branch was used.

#Note: the app must call this function

def fit_best_distribution(data: np.ndarray):
    discrete = is_discrete(data)
    results = fit_discrete(data) if discrete else fit_continuous(data)
    return results, discrete

# TESTING TO SHOW IT WORKS BELOW (DELETE LATER)


# **YULING'S PORTION (TASK 4)**

## **4. PARAMETERS**

#Create a Parameter Formatting Function

def extract_winning_parameters(results_df, discrete_flag):

    #Extracts and formats the parameters of the best-fitting (top-ranked) distribution.

    if results_df.empty or results_df.iloc[0]["params"] is None:
        return "No valid distribution fit found."

    best_row = results_df.iloc[0]
    dist_name = best_row["distribution"]
    params = best_row["params"]

    formatted_params = {}

    # Map parameters to readable labels depending on the distribution
    if dist_name == "Normal":
        formatted_params = {"Mean (loc)": params[0], "Standard Deviation (scale)": params[1]}
    elif dist_name == "Exponential":
        formatted_params = {"Shift (loc)": params[0], "Scale (1/rate)": params[1]}
    elif dist_name == "Gamma":
        formatted_params = {"Shape (a)": params[0], "Shift (loc)": params[1], "Scale": params[2]}
    elif dist_name == "Lognormal":
        formatted_params = {"Shape (s)": params[0], "Shift (loc)": params[1], "Scale": params[2]}
    elif dist_name == "Beta":
        formatted_params = {"Shape 1 (a)": params[0], "Shape 2 (b)": params[1], "Shift (loc)": params[2], "Scale": params[3]}
    elif dist_name == "Uniform":
        formatted_params = {
            "Minimum (loc)": params[0],
            "Maximum": params[0] + params[1],
            "Range (scale)": params[1]
        }
    elif dist_name == "Poisson":
        formatted_params = {"Lambda (mu)": params[0], "Shift (loc)": params[1]}
    elif dist_name == "Binomial":
        formatted_params = {"Trials (n)": params[0], "Probability (p)": params[1], "Shift (loc)": params[2]}
    elif dist_name == "Geometric":
        formatted_params = {"Probability (p)": params[0], "Shift (loc)": params[1]}
    elif dist_name == "Negative Binomial":
        formatted_params = {"Number of Successes (n)": params[0], "Probability (p)": params[1], "Shift (loc)": params[2]}
    elif dist_name == "Discrete Uniform":
        formatted_params = {
            "Minimum Value (low)": params[0],
            "Upper Bound (high, exclusive)": params[1],
            "Shift (loc)": params[2]
        }
    elif dist_name == "Bernoulli":
        formatted_params = {"Probability (p)": params[0], "Shift (loc)": params[1]}
    else:
        # Fallback for any other distribution
        formatted_params = {f"Param {i}": p for i, p in enumerate(params)}

    
# **MATTHIAS' PORTION (TASK 5)**

## **5. MOMENTS**

def calculate_moments(data: np.ndarray,
                      results_df: pd.DataFrame,
                      discrete_flag: bool):

    # Check that a valid winning distribution exists
    if results_df.empty or results_df.iloc[0]["params"] is None:
        raise ValueError("No valid winning distribution found.")

    # Get winning distribution
    best_row = results_df.iloc[0]
    dist_name = best_row["distribution"]
    params = tuple(best_row["params"])

    # Get correct scipy distribution
    if discrete_flag:
        dist = DISCRETE_CANDIDATES[dist_name]
    else:
        dist = CONTINUOUS_CANDIDATES[dist_name]

    # Recreate fitted distribution
    fitted_dist = dist(*params)


    # ==================================================
    # PART A: FIRST FOUR RAW MOMENTS
    # ==================================================

    raw_moment_rows = []

    for order in range(1, 5):

        # Empirical raw moment:
        # E[X^r] ≈ average of X^r
        empirical_moment = np.mean(data ** order)

        # Theoretical raw moment of winning distribution
        theoretical_moment = fitted_dist.moment(order)

        # Absolute difference
        difference = abs(
            empirical_moment - theoretical_moment
        )

        raw_moment_rows.append({
            "Moment": f"{order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} Raw Moment",
            "Raw Data": empirical_moment,
            "Winning Distribution": theoretical_moment,
            "Absolute Difference": difference
        })

    raw_moments_df = pd.DataFrame(raw_moment_rows)


    # ==================================================
    # PART B: DISTRIBUTION CHARACTERISTICS
    # ==================================================

    # Raw data statistics
    raw_mean = np.mean(data)
    raw_variance = np.var(data, ddof=0)
    raw_skewness = stats.skew(data, bias=True)
    raw_kurtosis = stats.kurtosis(
        data,
        fisher=False,
        bias=True
    )

    # Theoretical statistics from fitted distribution
    fitted_mean, fitted_variance, fitted_skewness, fitted_excess_kurtosis = \
        fitted_dist.stats(moments="mvsk")

    # scipy gives excess kurtosis, so add 3
    fitted_kurtosis = fitted_excess_kurtosis + 3


    characteristics_df = pd.DataFrame({

        "Statistic": [
            "Mean",
            "Variance",
            "Skewness",
            "Kurtosis"
        ],

        "Raw Data": [
            raw_mean,
            raw_variance,
            raw_skewness,
            raw_kurtosis
        ],

        "Winning Distribution": [
            float(fitted_mean),
            float(fitted_variance),
            float(fitted_skewness),
            float(fitted_kurtosis)
        ]
    })


    characteristics_df["Absolute Difference"] = abs(
        characteristics_df["Raw Data"]
        - characteristics_df["Winning Distribution"]
    )


    return dist_name, raw_moments_df, characteristics_df
