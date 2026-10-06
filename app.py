# -*- coding: utf-8 -*-
"""
BR2207 Distribution Fit Explorer - Streamlit front end.

Run with:  streamlit run app.py
The calculations live in distribution_fitter.py; this file only handles
input, layout and display.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from distribution_fitter import (
    MIN_OBSERVATIONS,
    calculate_moments,
    extract_winning_parameters,
    fit_best_distribution,
    get_fitted_distribution,
    load_from_paste,
    numeric_columns,
    read_table,
    validate_data,
)

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Distribution Fit Explorer", page_icon="📊", layout="wide")

st.title("📊 Distribution Fit Explorer")
st.write(
    "Upload your dataset or paste numerical values "
    "to find the best-fitting probability distribution."
)
st.info(
    "👋 **How to use:** Paste numerical values or upload a CSV/Excel file, "
    "then click **Analyse Data**. The app will automatically identify whether "
    "your data is continuous or discrete, test multiple probability "
    "distributions, and identify the best-fitting model."
)


# ============================================================
# 1. INPUT
# ============================================================

def get_input_data():
    """Render the input widgets. Returns (data, analyse_clicked); data is None if not ready."""
    st.header("1. Input Your Data")
    method = st.radio("How would you like to enter your data?", ["Paste Values", "Upload CSV/Excel"])

    raw_values = None

    if method == "Paste Values":
        pasted = st.text_area(
            "Paste your numerical values:",
            placeholder="Example: 12.5, 13.2, 14.8, 15.1...",
        )
        if pasted.strip():
            raw_values = ("paste", pasted)
    else:
        uploaded = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                df = read_table(uploaded, uploaded.name)
                columns = numeric_columns(df)
            except Exception as e:
                st.error(f"⚠️ {e}")
                return None, False

            # --- ADD PREVIEW HERE ---
            st.markdown(f"**Preview: {uploaded.name}**")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(
                "File loaded successfully. Select a numeric column below; "
                "blank and non-numeric cells will be handled automatically."
            )
            # ------------------------

            # The column selector sits above the button so changing it never clears the results
            column = columns[0]
            if len(columns) > 1:
                column = st.selectbox(
                    "Multiple numerical columns were found. Select the variable to analyse:",
                    columns,
                )
            raw_values = ("file", df[column].to_numpy())

    clicked = st.button("🔍 Analyse Data", type="primary")
    if not clicked:
        return None, False

    if raw_values is None:
        st.error("⚠️ Please enter values or upload a file before analysing.")
        return None, True

    try:
        kind, content = raw_values
        data = load_from_paste(content) if kind == "paste" else validate_data(content)
    except ValueError as e:
        st.error(f"⚠️ {e}")
        return None, True

    return data, True


# ============================================================
# DISPLAY SECTIONS
# ============================================================

def show_summary(winning_dist, discrete, n_obs):
    st.header("2. Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Fit", winning_dist)
    col2.metric("📊 Data Type", "Discrete" if discrete else "Continuous")
    col3.metric("🔢 Observations", n_obs)


def show_ranking(results, valid_results, discrete):
    st.header("3. Distribution Ranking")

    if discrete:
        columns = {"distribution": "Distribution", "aic": "AIC",
                   "chi2_stat": "Chi-Square Statistic", "chi2_pvalue": "p-value"}
    else:
        columns = {"distribution": "Distribution", "aic": "AIC",
                   "ks_stat": "KS Statistic", "ks_pvalue": "p-value"}

    table = valid_results[list(columns)].rename(columns=columns).round(4)
    table.insert(0, "Rank", range(1, len(table) + 1))

    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Distributions are ranked using AIC. A lower AIC indicates a better "
        "relative fit among the distributions tested."
    )

    # Never hide failed fits silently - show which ones failed and why
    failed = results[results["aic"].isna()]
    if not failed.empty:
        with st.expander(f"⚠️ {len(failed)} distribution(s) could not be fitted"):
            for _, row in failed.iterrows():
                st.write(f"**{row['distribution']}**: {row.get('error', 'unknown error')}")


def show_parameters(winning_dist, param_dict):
    st.header("4. Estimated Parameters")
    st.write(f"Estimated parameters for the **{winning_dist} distribution**:")

    cols = st.columns(len(param_dict))
    for col, (label, value) in zip(cols, param_dict.items()):
        col.metric(label, f"{float(value):.4f}")


def show_moments(data, results, discrete, winning_dist):
    st.header("5. Moments Comparison")
    st.write(
        f"Compare the moments of the raw data with those of the fitted "
        f"**{winning_dist} distribution**."
    )

    _, raw_moments_df, characteristics_df = calculate_moments(data, results, discrete)

    st.subheader("First Four Raw Moments")
    st.dataframe(raw_moments_df.round(4), hide_index=True, width="stretch")
    st.caption(
        "A smaller absolute difference means the fitted distribution more "
        "closely matches that moment of the observed data."
    )

    st.subheader("Distribution Characteristics")
    st.dataframe(characteristics_df.round(4), hide_index=True, width="stretch")
    st.caption(
        "These statistics compare the location, spread and shape of the "
        "raw data with the fitted distribution. Kurtosis is the Pearson "
        "definition (a normal distribution has kurtosis 3)."
    )


def plot_density(data, fitted, winning_dist, discrete):
    """Empirical distribution (histogram / bar chart) against the fitted PDF / PMF."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if discrete:
        x = np.arange(int(data.min()), int(data.max()) + 1)
        ax.bar(x, [np.mean(data == v) for v in x], alpha=0.6, label="Raw Data")
        ax.plot(x, fitted.pmf(x), marker="o", linewidth=2.5, label=f"Fitted {winning_dist}")
        ax.set_ylabel("Probability")
    else:
        ax.hist(data, bins="auto", density=True, alpha=0.6, label="Raw Data")
        lo, hi = data.min(), data.max()
        if hi == lo:
            x = np.linspace(lo - 1, hi + 1, 500)
        else:
            offset = (hi - lo) * 0.005  # avoid pdf singularities exactly at the edges
            x = np.linspace(lo + offset, hi - offset, 500)
        y = fitted.pdf(x)
        valid = np.isfinite(y) & (y >= 0)
        ax.plot(x[valid], y[valid], linewidth=2.5, label=f"Fitted {winning_dist}")
        ax.set_ylabel("Density")

    ax.set_title(f"Raw Data vs Fitted {winning_dist} Distribution")
    ax.set_xlabel("Observed Value")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_cdf(data, fitted, winning_dist, discrete):
    """Empirical CDF against the fitted CDF."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if discrete:
        x = np.arange(int(data.min()), int(data.max()) + 1)
        empirical = np.array([np.mean(data <= v) for v in x])
        marker = "o"
    else:
        x = np.sort(data)
        empirical = np.arange(1, len(x) + 1) / len(x)
        marker = None

    ax.step(x, empirical, where="post", linewidth=2, label="Empirical CDF")
    ax.plot(x, fitted.cdf(x), marker=marker, linewidth=2.5, label=f"Fitted {winning_dist} CDF")

    ax.set_title(f"Empirical CDF vs Fitted {winning_dist} CDF")
    ax.set_xlabel("Observed Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def show_visualisations(data, results, discrete):
    st.header("6. Data Visualization")
    winning_dist, fitted = get_fitted_distribution(results, discrete)
    st.write(f"Visual comparison of the raw data with the best-fitted **{winning_dist} distribution**.")

    st.subheader("Empirical vs Fitted Distribution")
    fig = plot_density(data, fitted, winning_dist, discrete)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The bars represent the empirical distribution of the observed data, "
        "while the fitted curve represents the best-fitting theoretical distribution."
    )

    st.subheader("Empirical CDF vs Fitted CDF")
    fig = plot_cdf(data, fitted, winning_dist, discrete)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The closer the empirical and fitted CDF lines are, the more closely "
        "the theoretical distribution matches the observed data."
    )


def show_interpretation(best, winning_dist, discrete):
    st.header("7. Interpretation")
    st.write("A simple summary of the distribution fitting results.")

    st.subheader("🏆 Best-Fitting Distribution")
    st.write(
        f"The **{winning_dist} distribution** was selected as the best fit because it "
        f"achieved the lowest AIC of **{best['aic']:.4f}** among the distributions tested."
    )

    st.subheader("📊 Goodness-of-Fit")
    if discrete:
        test_name, stat, p_value = "Chi-Square", best["chi2_stat"], best["chi2_pvalue"]
        explanation = (
            "A smaller Chi-Square statistic indicates that the fitted probabilities "
            "are closer to the observed frequencies."
        )
    else:
        test_name, stat, p_value = "Kolmogorov-Smirnov (KS)", best["ks_stat"], best["ks_pvalue"]
        explanation = (
            "The KS statistic measures the maximum difference between the empirical "
            "and fitted cumulative distributions. A smaller value indicates a closer fit."
        )

    if pd.notna(stat):
        st.write(f"The {test_name} statistic is **{stat:.4f}**, with a p-value of **{p_value:.4f}**.")
        st.caption(explanation)
    else:
        st.write("A goodness-of-fit test could not be computed for this dataset.")

    st.subheader("🔎 Statistical Interpretation")
    if pd.notna(p_value):
        if p_value >= 0.05:
            st.success(
                f"At the 5% significance level, the goodness-of-fit test does not provide "
                f"sufficient evidence to reject the {winning_dist} distribution."
            )
        else:
            st.warning(
                f"At the 5% significance level, the goodness-of-fit test provides evidence "
                f"against the {winning_dist} distribution. Although it ranks best among the "
                f"distributions tested, its absolute fit should be interpreted with caution."
            )
        if not discrete:
            st.caption(
                "Note: the KS p-value is approximate because the parameters were estimated "
                "from the same data, which tends to make the p-value slightly optimistic."
            )

    st.subheader("💡 Summary")
    st.info(
        f"Overall, **{winning_dist}** provides the best relative fit among the candidate "
        f"distributions tested. Review the ranking table, estimated parameters, moment "
        f"comparisons and visualizations above for a complete assessment of the fit."
    )


def show_glossary():
    with st.expander("ℹ️ Understanding the Results"):
        st.markdown(
            """
**AIC (Akaike Information Criterion)**  
Used to compare candidate distributions. A lower AIC indicates a better relative fit
among the distributions tested.

**KS Statistic**  
Used for continuous data. A smaller value indicates that the fitted cumulative
distribution is closer to the empirical data.

**Chi-Square Statistic**  
Used for discrete data. It measures differences between observed and expected frequencies.

**p-value**  
A small p-value, commonly below 0.05, provides evidence against the fitted distribution
under the goodness-of-fit test.

**Moments**  
Numerical measures describing characteristics of a distribution, including its location,
spread and shape.
            """
        )


# ============================================================
# MAIN
# ============================================================

data, analyse_clicked = get_input_data()

if data is not None:
    try:
        results, discrete = fit_best_distribution(data)

        # Failed fits have NaN AIC and are sorted last, so the top row is the winner
        valid_results = results.dropna(subset=["aic"])
        if valid_results.empty:
            st.error("⚠️ None of the candidate distributions could successfully fit this dataset.")
            st.stop()

        winning_dist, param_dict = extract_winning_parameters(results, discrete)

        st.success("✅ Analysis complete!")
        show_summary(winning_dist, discrete, len(data))
        show_ranking(results, valid_results, discrete)
        show_parameters(winning_dist, param_dict)
        show_moments(data, results, discrete, winning_dist)
        show_visualisations(data, results, discrete)
        show_interpretation(results.iloc[0], winning_dist, discrete)
        show_glossary()

    except Exception as e:
        st.error("⚠️ The analysis could not be completed. Please check your data and try again.")
        with st.expander("Technical details"):
            st.code(str(e))
