"""
hypothesis_tests.py
====================
A clean, well-documented library of hypothesis tests used throughout
this portfolio. Each function returns a structured dict with all
relevant statistics, making results easy to report and compare.

Tests implemented
-----------------
Z-tests  : one_sample_z, two_sample_z_proportions
T-tests  : one_sample_t, independent_t, paired_t, welch_t
Chi-square: chi_square_independence, chi_square_goodness_of_fit
Extras   : shapiro_normality, levene_variance, effect_size helpers
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x, decimals=4):
    """Format a float, replacing very small values with scientific notation."""
    if isinstance(x, float) and x < 0.0001:
        return f"{x:.2e}"
    return round(x, decimals) if isinstance(x, float) else x


def cohens_d(group1, group2):
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                          (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_one_sample(sample, mu0):
    """Cohen's d for one-sample test."""
    return (np.mean(sample) - mu0) / np.std(sample, ddof=1)


def cohens_d_paired(differences):
    """Cohen's d for paired test (d = mean_diff / std_diff)."""
    return np.mean(differences) / np.std(differences, ddof=1)


def cramers_v(chi2, n, r, c):
    """Cramer's V effect size for chi-square test."""
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"


def interpret_cramers_v(v, min_dim):
    """Interpret Cramer's V depending on minimum table dimension."""
    thresholds = {2: (0.10, 0.30, 0.50), 3: (0.07, 0.21, 0.35), 4: (0.06, 0.17, 0.29)}
    low, med, high = thresholds.get(min_dim, thresholds[4])
    if v < low:   return "negligible"
    if v < med:   return "small"
    if v < high:  return "medium"
    return "large"


def _result(test_name, stat, pval, alpha, extra=None):
    """Build a standard result dictionary."""
    reject = pval < alpha
    r = {
        'test'          : test_name,
        'statistic'     : _fmt(float(stat)),
        'p_value'       : _fmt(float(pval)),
        'alpha'         : alpha,
        'reject_H0'     : reject,
        'conclusion'    : ('Reject H₀ — statistically significant'
                           if reject else
                           'Fail to reject H₀ — not statistically significant'),
    }
    if extra:
        r.update(extra)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# ASSUMPTION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def shapiro_normality(data, name='sample', alpha=0.05):
    """
    Shapiro-Wilk normality test.
    H₀: data comes from a normal distribution.
    """
    # Shapiro-Wilk is limited to n ≤ 5000; sample if larger
    x = data.dropna().values if hasattr(data, 'dropna') else np.array(data)
    x = x[~np.isnan(x)]
    if len(x) > 5000:
        x = np.random.choice(x, 5000, replace=False)
    stat, pval = stats.shapiro(x)
    reject = pval < alpha
    return {
        'test'      : 'Shapiro-Wilk normality',
        'sample'    : name,
        'n'         : len(x),
        'W'         : _fmt(float(stat)),
        'p_value'   : _fmt(float(pval)),
        'normal'    : not reject,
        'conclusion': ('Data appears normal' if not reject
                       else 'Data departs from normality'),
    }


def levene_variance(group1, group2, alpha=0.05):
    """
    Levene's test for equality of variances.
    H₀: variances are equal (homoscedasticity).
    """
    stat, pval = stats.levene(group1, group2)
    reject = pval < alpha
    return {
        'test'        : "Levene's variance equality",
        'F_statistic' : _fmt(float(stat)),
        'p_value'     : _fmt(float(pval)),
        'equal_var'   : not reject,
        'conclusion'  : ('Equal variances assumed (use pooled T)'
                         if not reject else
                         'Unequal variances — use Welch T-test'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Z-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def one_sample_z(sample, mu0, sigma, alpha=0.05, tail='two'):
    """
    One-sample Z-test.

    Parameters
    ----------
    sample : array-like  — observed data
    mu0    : float       — hypothesized population mean (H₀)
    sigma  : float       — KNOWN population standard deviation
    alpha  : float       — significance level (default 0.05)
    tail   : str         — 'two', 'left', or 'right'

    H₀: μ = mu0
    H₁: μ ≠ mu0  (two-tailed)
    """
    x = np.array(sample)
    n = len(x)
    xbar = np.mean(x)
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se

    if tail == 'two':
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha/2)
        h1 = f'μ ≠ {mu0}'
    elif tail == 'right':
        pval = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1 - alpha)
        h1 = f'μ > {mu0}'
    else:
        pval = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        h1 = f'μ < {mu0}'

    ci_low = xbar - crit * se
    ci_high = xbar + crit * se

    return _result('One-sample Z-test', z, pval, alpha, {
        'H0'              : f'μ = {mu0}',
        'H1'              : h1,
        'tail'            : tail,
        'n'               : n,
        'sample_mean'     : _fmt(xbar),
        'known_sigma'     : sigma,
        'standard_error'  : _fmt(se),
        'critical_value'  : _fmt(crit if tail != 'left' else -crit),
        f'CI_{int((1-alpha)*100)}pct': f'({_fmt(ci_low)}, {_fmt(ci_high)})',
        'effect_size_d'   : _fmt(cohens_d_one_sample(x, mu0)),
    })


def two_sample_z_proportions(n1, x1, n2, x2, alpha=0.05, tail='two'):
    """
    Two-sample Z-test for proportions.

    Parameters
    ----------
    n1, n2 : int   — sample sizes
    x1, x2 : int   — number of successes in each group

    H₀: p1 = p2
    """
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se

    if tail == 'two':
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha/2)
    elif tail == 'right':
        pval = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1 - alpha)
    else:
        pval = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)

    # 95% CI for difference in proportions
    se_ci = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    ci_low  = (p1-p2) - stats.norm.ppf(1-alpha/2)*se_ci
    ci_high = (p1-p2) + stats.norm.ppf(1-alpha/2)*se_ci

    return _result('Two-sample Z-test (proportions)', z, pval, alpha, {
        'H0'               : 'p₁ = p₂',
        'H1'               : 'p₁ ≠ p₂' if tail=='two' else f'p₁ {">" if tail=="right" else "<"} p₂',
        'tail'             : tail,
        'group1'           : {'n': n1, 'successes': x1, 'proportion': _fmt(p1)},
        'group2'           : {'n': n2, 'successes': x2, 'proportion': _fmt(p2)},
        'pooled_proportion': _fmt(p_pool),
        'proportion_diff'  : _fmt(p1-p2),
        'relative_lift_pct': _fmt((p1-p2)/p2*100),
        'critical_value'   : _fmt(crit),
        f'CI_{int((1-alpha)*100)}pct_diff': f'({_fmt(ci_low)}, {_fmt(ci_high)})',
    })


# ─────────────────────────────────────────────────────────────────────────────
# T-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def one_sample_t(sample, mu0, alpha=0.05, tail='two'):
    """
    One-sample T-test.

    H₀: μ = mu0
    Use when σ is unknown (estimated from sample).
    """
    x = np.array(sample)
    n = len(x)
    xbar = np.mean(x)
    s = np.std(x, ddof=1)
    se = s / np.sqrt(n)
    df = n - 1

    if tail == 'two':
        stat, pval = stats.ttest_1samp(x, mu0)
        crit = stats.t.ppf(1 - alpha/2, df)
        h1 = f'μ ≠ {mu0}'
    elif tail == 'right':
        stat, pval_two = stats.ttest_1samp(x, mu0)
        pval = pval_two / 2 if stat > 0 else 1 - pval_two/2
        crit = stats.t.ppf(1 - alpha, df)
        h1 = f'μ > {mu0}'
    else:
        stat, pval_two = stats.ttest_1samp(x, mu0)
        pval = pval_two / 2 if stat < 0 else 1 - pval_two/2
        crit = stats.t.ppf(alpha, df)
        h1 = f'μ < {mu0}'

    ci = stats.t.interval(1-alpha, df, loc=xbar, scale=se)

    return _result('One-sample T-test', stat, pval, alpha, {
        'H0'             : f'μ = {mu0}',
        'H1'             : h1,
        'tail'           : tail,
        'n'              : n,
        'df'             : df,
        'sample_mean'    : _fmt(xbar),
        'sample_std'     : _fmt(s),
        'standard_error' : _fmt(se),
        'critical_value' : _fmt(crit),
        f'CI_{int((1-alpha)*100)}pct': f'({_fmt(ci[0])}, {_fmt(ci[1])})',
        'effect_size_d'  : _fmt(cohens_d_one_sample(x, mu0)),
        'effect_magnitude': interpret_cohens_d(cohens_d_one_sample(x, mu0)),
    })


def independent_t(group1, group2, alpha=0.05, tail='two', equal_var=None):
    """
    Independent two-sample T-test (pooled or Welch).

    If equal_var=None, Levene's test is run automatically to decide.

    H₀: μ₁ = μ₂
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    # Auto-check variance equality
    if equal_var is None:
        lev = levene_variance(g1, g2, alpha)
        equal_var = lev['equal_var']
        test_type = 'Pooled T-test' if equal_var else "Welch's T-test"
    else:
        test_type = 'Pooled T-test' if equal_var else "Welch's T-test"

    stat, pval_two = stats.ttest_ind(g1, g2, equal_var=equal_var)
    df = len(g1)+len(g2)-2 if equal_var else None   # Welch df is approximate

    if tail == 'two':
        pval = pval_two
    elif tail == 'right':
        pval = pval_two/2 if stat > 0 else 1-pval_two/2
    else:
        pval = pval_two/2 if stat < 0 else 1-pval_two/2

    d = cohens_d(g1, g2)
    diff = np.mean(g1) - np.mean(g2)
    se = np.sqrt(np.var(g1,ddof=1)/len(g1) + np.var(g2,ddof=1)/len(g2))
    crit = stats.norm.ppf(1-alpha/2) if (len(g1)+len(g2)) > 60 else stats.t.ppf(1-alpha/2, len(g1)+len(g2)-2)

    return _result(test_type, stat, pval, alpha, {
        'H0'                 : 'μ₁ = μ₂',
        'H1'                 : 'μ₁ ≠ μ₂' if tail=='two' else f'μ₁ {">" if tail=="right" else "<"} μ₂',
        'tail'               : tail,
        'equal_var_assumed'  : equal_var,
        'group1'             : {'n': len(g1), 'mean': _fmt(np.mean(g1)), 'std': _fmt(np.std(g1,ddof=1))},
        'group2'             : {'n': len(g2), 'mean': _fmt(np.mean(g2)), 'std': _fmt(np.std(g2,ddof=1))},
        'mean_difference'    : _fmt(diff),
        'effect_size_d'      : _fmt(d),
        'effect_magnitude'   : interpret_cohens_d(d),
        'critical_value'     : _fmt(crit),
        'power_note'         : 'Run power analysis to confirm adequate sample size',
    })


def paired_t(before, after, alpha=0.05, tail='two'):
    """
    Paired (dependent) T-test.

    H₀: μ_diff = 0  (no change)
    Use when the same subjects are measured twice.
    """
    b = np.array(before)
    a = np.array(after)
    diff = a - b
    n = len(diff)
    df = n - 1

    stat, pval_two = stats.ttest_rel(b, a)
    stat = -stat   # flip: (after - before)

    if tail == 'two':
        pval = pval_two
    elif tail == 'right':
        pval = pval_two/2 if stat > 0 else 1-pval_two/2
    else:
        pval = pval_two/2 if stat < 0 else 1-pval_two/2

    d = cohens_d_paired(diff)
    ci = stats.t.interval(1-alpha, df, loc=np.mean(diff), scale=stats.sem(diff))

    return _result('Paired T-test', stat, pval, alpha, {
        'H0'                : 'μ_difference = 0',
        'H1'                : 'μ_difference ≠ 0' if tail=='two' else ('μ_after > μ_before' if tail=='right' else 'μ_after < μ_before'),
        'tail'              : tail,
        'n_pairs'           : n,
        'df'                : df,
        'mean_before'       : _fmt(np.mean(b)),
        'mean_after'        : _fmt(np.mean(a)),
        'mean_difference'   : _fmt(np.mean(diff)),
        'std_difference'    : _fmt(np.std(diff, ddof=1)),
        f'CI_{int((1-alpha)*100)}pct_diff': f'({_fmt(ci[0])}, {_fmt(ci[1])})',
        'effect_size_d'     : _fmt(d),
        'effect_magnitude'  : interpret_cohens_d(d),
    })


def welch_t(group1, group2, alpha=0.05, tail='two'):
    """Welch's T-test — for unequal variances. Alias for independent_t with equal_var=False."""
    return independent_t(group1, group2, alpha=alpha, tail=tail, equal_var=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHI-SQUARE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def chi_square_independence(data, col1, col2, alpha=0.05):
    """
    Chi-square test of independence.

    H₀: col1 and col2 are independent (no association).
    H₁: col1 and col2 are associated.

    Parameters
    ----------
    data : pd.DataFrame
    col1, col2 : str — column names for the two categorical variables
    """
    ct = pd.crosstab(data[col1], data[col2])
    chi2, pval, dof, expected = stats.chi2_contingency(ct)

    n = ct.values.sum()
    r, c = ct.shape
    v = cramers_v(chi2, n, r, c)

    # Check assumption: all expected counts ≥ 5
    low_expected = (expected < 5).sum()
    assumption_ok = low_expected == 0

    return _result("Chi-square test of independence", chi2, pval, alpha, {
        'H0'                : f'"{col1}" and "{col2}" are independent',
        'H1'                : f'"{col1}" and "{col2}" are associated',
        'df'                : dof,
        'table_shape'       : f'{r} rows × {c} cols',
        'n_total'           : int(n),
        'critical_value'    : _fmt(stats.chi2.ppf(1-alpha, dof)),
        'cramers_v'         : _fmt(v),
        'effect_magnitude'  : interpret_cramers_v(v, min(r,c)),
        'contingency_table' : ct.to_dict(),
        'assumption_ok'     : assumption_ok,
        'cells_below_5'     : int(low_expected),
        'assumption_note'   : ('All expected counts ≥ 5 ✓' if assumption_ok
                               else f'Warning: {low_expected} cell(s) have expected count < 5. Consider Fisher\'s exact test or collapsing categories.'),
    })


def chi_square_goodness_of_fit(observed, expected_probs, categories=None, alpha=0.05):
    """
    Chi-square goodness-of-fit test.

    H₀: observed frequencies match the expected distribution.

    Parameters
    ----------
    observed       : array-like — observed counts
    expected_probs : array-like — expected probabilities (must sum to 1)
    categories     : list       — category labels (optional)
    """
    obs = np.array(observed)
    probs = np.array(expected_probs)
    n = obs.sum()
    exp = n * probs
    dof = len(obs) - 1

    chi2, pval = stats.chisquare(obs, f_exp=exp)

    low_exp = (exp < 5).sum()

    result_data = {
        'H0'            : 'Observed frequencies match expected distribution',
        'H1'            : 'Observed frequencies differ from expected',
        'df'            : dof,
        'n_total'       : int(n),
        'critical_value': _fmt(stats.chi2.ppf(1-alpha, dof)),
        'cells_below_5' : int(low_exp),
        'assumption_ok' : low_exp == 0,
    }
    if categories is not None:
        result_data['breakdown'] = {
            cat: {'observed': int(o), 'expected': round(float(e), 2), 'contribution': _fmt((o-e)**2/e)}
            for cat, o, e in zip(categories, obs, exp)
        }
    return _result('Chi-square goodness-of-fit', chi2, pval, alpha, result_data)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result, title=None):
    """Pretty-print a test result dict."""
    sep = '─' * 60
    print(f'\n{sep}')
    if title:
        print(f'  {title}')
    print(f'  Test     : {result["test"]}')
    print(f'  H₀       : {result.get("H0", "—")}')
    print(f'  H₁       : {result.get("H1", "—")}')
    print(f'  Statistic: {result["statistic"]}')
    print(f'  p-value  : {result["p_value"]}')
    print(f'  α        : {result["alpha"]}')
    print(f'  Decision : {"✗ " + result["conclusion"] if result["reject_H0"] else "✓ " + result["conclusion"]}')
    for k, v in result.items():
        if k not in ('test','H0','H1','statistic','p_value','alpha','reject_H0','conclusion','contingency_table'):
            if not isinstance(v, dict):
                print(f'  {k:<20}: {v}')
    print(sep)
