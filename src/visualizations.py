"""
visualizations.py
=================
Publication-quality plots for hypothesis testing results.
Each function saves a figure and returns the filepath.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
NAVY   = '#1A3A5C'
BLUE   = '#185FA5'
TEAL   = '#0F6E56'
CORAL  = '#993C1D'
AMBER  = '#C17A1A'
GRAY   = '#5F5E5A'
LIGHT  = '#F1EFE8'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor'  : '#FAFAF8',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'grid.linestyle'  : '--',
    'font.family'     : 'DejaVu Sans',
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
def plot_z_distribution(z_stat, alpha=0.05, tail='two', title='Z-test', save_path=None):
    """Plot Z-distribution with critical region and test statistic."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    z_stat = float(z_stat)
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)

    ax.plot(x, y, color=BLUE, lw=2.5, label='Standard Normal N(0,1)')
    ax.fill_between(x, y, alpha=0.08, color=BLUE)

    if tail == 'two':
        crit = stats.norm.ppf(1 - alpha/2)
        mask_l = x <= -crit
        mask_r = x >= crit
        ax.fill_between(x[mask_l], y[mask_l], color=CORAL, alpha=0.55, label=f'Rejection region (α={alpha})')
        ax.fill_between(x[mask_r], y[mask_r], color=CORAL, alpha=0.55)
        ax.axvline(-crit, color=CORAL, lw=1.5, ls='--', label=f'Critical value ±{crit:.3f}')
        ax.axvline( crit, color=CORAL, lw=1.5, ls='--')
    elif tail == 'right':
        crit = stats.norm.ppf(1 - alpha)
        mask = x >= crit
        ax.fill_between(x[mask], y[mask], color=CORAL, alpha=0.55, label=f'Rejection region (α={alpha})')
        ax.axvline(crit, color=CORAL, lw=1.5, ls='--', label=f'Critical value {crit:.3f}')
    else:
        crit = stats.norm.ppf(alpha)
        mask = x <= crit
        ax.fill_between(x[mask], y[mask], color=CORAL, alpha=0.55, label=f'Rejection region (α={alpha})')
        ax.axvline(crit, color=CORAL, lw=1.5, ls='--', label=f'Critical value {crit:.3f}')

    ax.axvline(z_stat, color=AMBER, lw=2.5, label=f'Z statistic = {z_stat:.3f}')
    pval = 2*(1-stats.norm.cdf(abs(z_stat))) if tail=='two' else (1-stats.norm.cdf(z_stat) if tail=='right' else stats.norm.cdf(z_stat))
    rejected = pval < alpha

    ax.set_title(f'{title}\n{"✗ Reject H₀" if rejected else "✓ Fail to reject H₀"} — p-value = {pval:.4f}',
                 fontweight='bold', color=CORAL if rejected else TEAL, pad=12)
    ax.set_xlabel('Z score')
    ax.set_ylabel('Probability density')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_t_distribution(t_stat, df, alpha=0.05, tail='two', title='T-test', save_path=None):
    """Plot T-distribution with normal overlay for comparison."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t_stat = float(t_stat)
    lim = max(4, abs(t_stat) + 1)
    x = np.linspace(-lim, lim, 500)

    y_t    = stats.t.pdf(x, df)
    y_norm = stats.norm.pdf(x)
    crit   = stats.t.ppf(1 - alpha/2, df)

    ax.plot(x, y_norm, color=GRAY, lw=1.5, ls=':', label='Normal (for reference)', alpha=0.6)
    ax.plot(x, y_t,   color=BLUE, lw=2.5, label=f'T-distribution (df={df})')

    if tail == 'two':
        ax.fill_between(x, y_t, where=(x <= -crit), color=CORAL, alpha=0.45, label=f'Rejection (α={alpha})')
        ax.fill_between(x, y_t, where=(x >= crit),  color=CORAL, alpha=0.45)
        ax.axvline(-crit, color=CORAL, ls='--', lw=1.5, label=f'±{crit:.3f}')
        ax.axvline( crit, color=CORAL, ls='--', lw=1.5)
        pval = 2 * stats.t.sf(abs(t_stat), df)
    elif tail == 'right':
        ax.fill_between(x, y_t, where=(x >= crit), color=CORAL, alpha=0.45, label=f'Rejection (α={alpha})')
        ax.axvline(crit, color=CORAL, ls='--', lw=1.5)
        pval = stats.t.sf(t_stat, df)
    else:
        ax.fill_between(x, y_t, where=(x <= -crit), color=CORAL, alpha=0.45, label=f'Rejection (α={alpha})')
        ax.axvline(-crit, color=CORAL, ls='--', lw=1.5)
        pval = stats.t.cdf(t_stat, df)

    ax.axvline(t_stat, color=AMBER, lw=2.5, label=f't statistic = {t_stat:.3f}')
    rejected = pval < alpha
    ax.set_title(f'{title} — df={df}\n{"✗ Reject H₀" if rejected else "✓ Fail to reject H₀"} — p-value = {pval:.4f}',
                 fontweight='bold', color=CORAL if rejected else TEAL)
    ax.set_xlabel('t score')
    ax.set_ylabel('Probability density')
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_chi_distribution(chi2_stat, df, alpha=0.05, title='Chi-square test', save_path=None):
    """Plot Chi-square distribution with critical region."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    chi2_stat = float(chi2_stat)
    x_max = max(chi2_stat * 1.6, stats.chi2.ppf(0.999, df))
    x = np.linspace(0.001, x_max, 500)
    y = stats.chi2.pdf(x, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    pval = 1 - stats.chi2.cdf(chi2_stat, df)

    ax.plot(x, y, color=CORAL, lw=2.5, label=f'χ²-distribution (df={df})')
    ax.fill_between(x, y, alpha=0.08, color=CORAL)
    ax.fill_between(x, y, where=(x >= crit), color=CORAL, alpha=0.45, label=f'Rejection (α={alpha})')
    ax.axvline(crit,      color=CORAL, ls='--', lw=1.5, label=f'Critical value {crit:.3f}')
    ax.axvline(chi2_stat, color=AMBER, lw=2.5, label=f'χ² statistic = {chi2_stat:.3f}')

    rejected = pval < alpha
    ax.set_title(f'{title} — df={df}\n{"✗ Reject H₀" if rejected else "✓ Fail to reject H₀"} — p-value = {pval:.4f}',
                 fontweight='bold', color=CORAL if rejected else TEAL)
    ax.set_xlabel('χ² value')
    ax.set_ylabel('Probability density')
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_distribution_shapes(save_path=None):
    """Side-by-side comparison of T and Chi-square distribution shapes at various df."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # T-distribution
    ax = axes[0]
    x = np.linspace(-5, 5, 400)
    df_values = [1, 2, 5, 15, 30]
    colors_t = [CORAL, AMBER, TEAL, BLUE, NAVY]
    for df, col in zip(df_values, colors_t):
        ax.plot(x, stats.t.pdf(x, df), color=col, lw=2, label=f'df={df}')
    ax.plot(x, stats.norm.pdf(x), color=GRAY, lw=1.5, ls='--', label='Normal (df→∞)')
    ax.set_title("T-distribution by degrees of freedom\n(heavier tails → converges to normal)", fontweight='bold')
    ax.set_xlabel('t value'); ax.set_ylabel('Density')
    ax.legend(fontsize=9); ax.set_xlim(-5, 5); ax.set_ylim(0, 0.42)
    ax.annotate('Fat tails at\nlow df', xy=(-4, 0.06), fontsize=8, color=CORAL, ha='center')
    ax.annotate('Approaches\nnormal', xy=(3.5, 0.37), fontsize=8, color=NAVY, ha='center')

    # Chi-square distribution
    ax = axes[1]
    k_values = [1, 2, 3, 5, 10, 20]
    colors_c = [CORAL, AMBER, '#CC7A00', TEAL, BLUE, NAVY]
    for k, col in zip(k_values, colors_c):
        x_c = np.linspace(0.01, 40, 500)
        ax.plot(x_c, stats.chi2.pdf(x_c, k), color=col, lw=2, label=f'k={k}')
    ax.set_title("Chi-square distribution by degrees of freedom\n(right-skewed → approaches normal)", fontweight='bold')
    ax.set_xlabel('χ² value'); ax.set_ylabel('Density')
    ax.set_xlim(0, 40); ax.set_ylim(0, 0.5)
    ax.legend(fontsize=9)
    ax.annotate('J-shape\n(k=1,2)', xy=(1, 0.45), fontsize=8, color=CORAL)
    ax.annotate('Skewed\nhump', xy=(8, 0.13), fontsize=8, color=TEAL)

    fig.suptitle('Probability Distribution Shapes: T vs Chi-square', fontsize=14, fontweight='bold', color=NAVY, y=1.02)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_ab_test(df, metric_col, group_col, alpha=0.05, save_path=None):
    """Comprehensive A/B test visualization: distributions, means, conversion."""
    # Clean string matching: normalize column values to lowercase for checking
    group_series = df[group_col].astype(str).str.lower()
    
    # Dynamically match 'control' or 'placebo' for the baseline group
    is_ctrl = group_series.isin(['control', 'placebo'])
    # Match 'treatment', 'drug', or whatever else is the variant
    is_trt = group_series.isin(['treatment', 'drug', 'variant', 'test'])
    
    # If standard names aren't used, fall back to the two unique values present
    unique_groups = df[group_col].dropna().unique()
    if not is_ctrl.any() and len(unique_groups) >= 2:
        ctrl_name, trt_name = unique_groups[0], unique_groups[1]
        ctrl = df[df[group_col] == ctrl_name][metric_col].dropna()
        trt = df[df[group_col] == trt_name][metric_col].dropna()
    else:
        ctrl = df[is_ctrl][metric_col].dropna()
        trt = df[is_trt][metric_col].dropna()
        ctrl_name, trt_name = 'Control', 'Treatment'

    # --- Leave the rest of your visual code exactly as it is ---

#    ctrl = df[df[group_col]=='drug'][metric_col].dropna()
#    trt  = df[df[group_col]=='placebo'][metric_col].dropna()

    # --- DEBUGGING HEALTH CHECK ---
    print(f"Control group size: {len(ctrl)}, Variance: {ctrl.var():.4f}")
    print(f"Treatment group size: {len(trt)}, Variance: {trt.var():.4f}")

    if len(ctrl) < 2 or len(trt) < 2:
        raise ValueError("A/B test failed: One of your groups has fewer than 2 data points.")
        
    if ctrl.var() == 0 and trt.var() == 0:
        print("Warning: Both groups have 0 variance. P-value cannot be calculated via t-test.")
        # Optional fallback: manually set pval if means are different or identical
        pval = 0.0 if ctrl.mean() != trt.mean() else 1.0

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Distribution overlay
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(ctrl, bins=40, alpha=0.5, color=BLUE, label='Control', density=True)
    ax1.hist(trt,  bins=40, alpha=0.5, color=TEAL, label='Treatment', density=True)
    ax1.axvline(ctrl.mean(), color=BLUE, lw=2, ls='--', label=f'ctrl mean={ctrl.mean():.1f}')
    ax1.axvline(trt.mean(),  color=TEAL, lw=2, ls='--', label=f'trt mean={trt.mean():.1f}')
    ax1.set_title(f'Distribution of {metric_col}', fontweight='bold')
    ax1.set_xlabel(metric_col); ax1.legend(fontsize=8)

    # Box plots
    ax2 = fig.add_subplot(gs[1])
    bp = ax2.boxplot([ctrl, trt],
                     patch_artist=True, notch=True,
                     boxprops=dict(alpha=0.6))
    ax2.set_xticklabels(['Control', 'Treatment'])
    bp['boxes'][0].set_facecolor(BLUE)
    bp['boxes'][1].set_facecolor(TEAL)
    ax2.set_title('Box plots with notched CI', fontweight='bold')
    ax2.set_ylabel(metric_col)

    # Means with CI
    ax3 = fig.add_subplot(gs[2])
    means = [ctrl.mean(), trt.mean()]
    sems  = [stats.sem(ctrl), stats.sem(trt)]
    cis   = [1.96*s for s in sems]
    colors = [BLUE, TEAL]
    bars = ax3.bar(['Control','Treatment'], means, color=colors, alpha=0.7, width=0.5)
    ax3.errorbar(['Control','Treatment'], means, yerr=cis,
                 fmt='none', color='black', capsize=6, lw=2)
    ax3.set_title('Means with 95% CI', fontweight='bold')
    ax3.set_ylabel(f'Mean {metric_col}')
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(cis)*0.1,
                 f'{m:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    stat, pval = stats.ttest_ind(ctrl, trt, equal_var=False)
    print("pval", pval)
    # Corrected text flags
    sig_text = "(Significant ✓)" if pval < alpha else "(Not Significant ✗)"

    title_color = CORAL if pval < alpha else TEAL
    
    fig.suptitle(f'A/B Test: {metric_col} — p={pval:.12f} {sig_text}',
                 fontsize=13, fontweight='bold', color=title_color)

#    fig.suptitle(f'A/B Test: {metric_col} — p={pval:.4f} {"(significant ✗)" if pval<alpha else "(not significant ✓)"}',
#                 fontsize=13, fontweight='bold', color=CORAL if pval<alpha else TEAL)
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_paired_test(before, after, labels=('Before','After'), save_path=None):
    """Visualize paired data: individual lines + difference distribution."""
    diff = np.array(after) - np.array(before)
    n = len(diff)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Individual lines
    ax = axes[0]
    for i, (b, a) in enumerate(zip(before, after)):
        color = TEAL if a > b else CORAL
        ax.plot([0, 1], [b, a], color=color, alpha=0.3, lw=1)
    ax.plot([0,1], [np.mean(before), np.mean(after)], color=NAVY, lw=3, label='Group mean')
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_title('Individual changes\n(green=increase, red=decrease)', fontweight='bold')
    ax.set_ylabel('Score')

    # Difference histogram
    ax = axes[1]
    ax.hist(diff, bins=20, color=BLUE, alpha=0.7, edgecolor='white')
    ax.axvline(0,          color=GRAY,  lw=2, ls='--', label='No change')
    ax.axvline(diff.mean(),color=AMBER, lw=2.5, label=f'Mean diff = {diff.mean():.2f}')
    ax.set_title('Distribution of differences\n(after − before)', fontweight='bold')
    ax.set_xlabel('Difference'); ax.legend(fontsize=9)

    # Q-Q plot of differences
    ax = axes[2]
    qq = stats.probplot(diff, dist="norm")
    ax.scatter(qq[0][0], qq[0][1], color=BLUE, alpha=0.6, s=20)
    ax.plot(qq[0][0], qq[1][0]*qq[0][0]+qq[1][1], color=CORAL, lw=2, label='Normal line')
    ax.set_title('Q-Q plot of differences\n(checking normality assumption)', fontweight='bold')
    ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')
    ax.legend(fontsize=9)

    stat, pval = stats.ttest_rel(before, after)
    fig.suptitle(f'Paired T-test — p={pval:.4f} {"(significant ✗)" if pval<0.05 else "(✓)"}',
                 fontsize=13, fontweight='bold', color=CORAL if pval<0.05 else TEAL)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_contingency_heatmap(data, col1, col2, save_path=None):
    """Heatmap of observed vs expected counts for chi-square test."""
    ct = pd.crosstab(data[col1], data[col2])
    chi2, pval, dof, expected = stats.chi2_contingency(ct)
    residuals = (ct.values - expected) / np.sqrt(expected)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def heatmap(ax, data_arr, row_labels, col_labels, title, cmap, fmt='.0f', vmin=None, vmax=None):
        im = ax.imshow(data_arr, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontweight='bold', pad=10)
        for i in range(data_arr.shape[0]):
            for j in range(data_arr.shape[1]):
                ax.text(j, i, f'{data_arr[i,j]:{fmt}}', ha='center', va='center',
                        fontsize=9, color='white' if abs(data_arr[i,j]) > data_arr.max()*0.6 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    rows, cols = ct.index.astype(str).tolist(), ct.columns.astype(str).tolist()
    heatmap(axes[0], ct.values.astype(float), rows, cols, 'Observed counts', 'Blues')
    heatmap(axes[1], expected, rows, cols, 'Expected counts (under H₀)', 'Greens', fmt='.1f')
    v = max(abs(residuals.min()), abs(residuals.max()))
    heatmap(axes[2], residuals, rows, cols, 'Standardized residuals\n(|>2| = significant cell)', 'RdBu_r', fmt='.2f', vmin=-v, vmax=v)
    axes[0].set_xlabel(col2); axes[0].set_ylabel(col1)
    axes[1].set_xlabel(col2); axes[2].set_xlabel(col2)

    fig.suptitle(f'Chi-square Test: {col1} × {col2}\n'
                 f'χ²={chi2:.3f}, df={dof}, p={pval:.4f} '
                 f'— {"✗ Reject H₀ (associated)" if pval<0.05 else "✓ Fail to reject H₀ (independent)"}',
                 fontsize=12, fontweight='bold', color=CORAL if pval<0.05 else TEAL)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig


def plot_assumption_checks(data, group_col, value_col, save_path=None):
    """Check normality and variance assumptions for T-tests."""
    groups = data[group_col].unique()
    n_groups = len(groups)
    fig, axes = plt.subplots(2, n_groups, figsize=(5*n_groups, 8))
    if n_groups == 1:
        axes = axes.reshape(-1, 1)

    colors = [BLUE, TEAL, CORAL, AMBER]
    for j, (grp, col) in enumerate(zip(groups, colors)):
        vals = data[data[group_col]==grp][value_col].dropna()
        # Histogram + normal curve
        ax = axes[0, j]
        ax.hist(vals, bins=25, density=True, color=col, alpha=0.6, edgecolor='white')
        mu, sigma = vals.mean(), vals.std()
        x_line = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x_line, stats.norm.pdf(x_line, mu, sigma), color=NAVY, lw=2, label='Normal fit')
        stat_sw, p_sw = stats.shapiro(vals[:5000])
        ax.set_title(f'{grp} (n={len(vals)})\nShapiro-Wilk p={p_sw:.4f} — {"Normal ✓" if p_sw>0.05 else "Non-normal ✗"}',
                     fontweight='bold', color=TEAL if p_sw>0.05 else CORAL)
        ax.set_xlabel(value_col); ax.legend(fontsize=8)

        # Q-Q plot
        ax = axes[1, j]
        qq = stats.probplot(vals, dist='norm')
        ax.scatter(qq[0][0], qq[0][1], color=col, alpha=0.5, s=15)
        ax.plot(qq[0][0], qq[1][0]*qq[0][0]+qq[1][1], color=CORAL, lw=2)
        ax.set_title(f'Q-Q Plot: {grp}', fontweight='bold')
        ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')

    fig.suptitle(f'Assumption Checks: Normality of {value_col} by {group_col}',
                 fontsize=13, fontweight='bold', color=NAVY, y=1.01)
    fig.tight_layout()
    plt.show()
    return _save(fig, save_path) if save_path else fig
