"""
run_analysis.py
===============
End-to-end hypothesis testing across 5 real-world datasets.
Runs all Z-tests, T-tests, and Chi-square tests with full
assumption checking, effect sizes, and visualizations.

Run: python3 run_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from src.hypothesis_tests import (
    one_sample_z, two_sample_z_proportions,
    one_sample_t, independent_t, paired_t, welch_t,
    chi_square_independence, chi_square_goodness_of_fit,
    shapiro_normality, levene_variance, print_result
)
from src.visualizations import (
    plot_z_distribution, plot_t_distribution, plot_chi_distribution,
    plot_distribution_shapes, plot_ab_test, plot_paired_test,
    plot_contingency_heatmap, plot_assumption_checks
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')

print("\n" + "═"*65)
print("  HYPOTHESIS TESTING PORTFOLIO — FULL ANALYSIS")
print("═"*65)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: Manufacturing — One-Sample Z-test & T-test
# "Is the average product weight = 500g (target)?"
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  DATASET 1: Manufacturing Quality Control")
print("   File: data/manufacturing.csv")
print("   Goal: Test if machine output meets 500g weight specification\n")

mfg = pd.read_csv(f'{DATA_DIR}/manufacturing.csv')
weights = mfg['weight_g']
print(f"   Sample stats: n={len(weights)}, mean={weights.mean():.3f}g, std={weights.std(ddof=1):.3f}g")

# Assumption check
norm_check = shapiro_normality(weights, 'weight_g')
print(f"\n   Normality (Shapiro-Wilk): W={norm_check['W']}, p={norm_check['p_value']} — {norm_check['conclusion']}")

# One-sample Z-test (σ known from historical data = 3.5g)
print("\n   ── Test 1a: One-sample Z-test (σ=3.5 known from historical data)")
z_result = one_sample_z(weights, mu0=500, sigma=3.5, alpha=0.05, tail='two')
print_result(z_result, "Is mean weight = 500g? (Z-test, σ known)")
plot_z_distribution(z_result['statistic'], title='Manufacturing: Weight Z-test',
                    save_path=f'{RESULTS_DIR}/01_manufacturing_z_test.png')

# One-sample T-test (σ unknown, estimated from data)
print("\n   ── Test 1b: One-sample T-test (σ unknown, estimated from sample)")
t_result = one_sample_t(weights, mu0=500, alpha=0.05, tail='two')
print_result(t_result, "Is mean weight = 500g? (T-test, σ unknown)")
plot_t_distribution(t_result['statistic'], df=len(weights)-1, title='Manufacturing: Weight T-test',
                    save_path=f'{RESULTS_DIR}/02_manufacturing_t_test.png')

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: Clinical Trial — Independent Two-Sample T-test & Welch's T-test
# "Does the drug significantly reduce blood pressure vs placebo?"
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  DATASET 2: Clinical Trial — Drug vs Placebo")
print("   File: data/clinical_trial.csv")
print("   Goal: Does the drug reduce blood pressure more than placebo?\n")

clinical = pd.read_csv(f'{DATA_DIR}/clinical_trial.csv')
drug_bp    = clinical[clinical['group']=='drug']['bp_reduction']
placebo_bp = clinical[clinical['group']=='placebo']['bp_reduction']

print(f"   Drug:    n={len(drug_bp)}, mean={drug_bp.mean():.2f}, std={drug_bp.std():.2f}")
print(f"   Placebo: n={len(placebo_bp)}, mean={placebo_bp.mean():.2f}, std={placebo_bp.std():.2f}")

# Assumption checks
print("\n   ── Assumption Checks ──")
for grp, data in [('drug', drug_bp), ('placebo', placebo_bp)]:
    n = shapiro_normality(data, grp)
    print(f"   Normality [{grp}]: p={n['p_value']} — {n['conclusion']}")
lev = levene_variance(drug_bp, placebo_bp)
print(f"   Levene's test: p={lev['p_value']} — {lev['conclusion']}")

plot_assumption_checks(clinical, 'group', 'bp_reduction',
                       save_path=f'{RESULTS_DIR}/03_clinical_assumption_checks.png')

# Pooled T-test
print("\n   ── Test 2a: Independent T-test (pooled, equal variance)")
ind_result = independent_t(drug_bp, placebo_bp, alpha=0.05, equal_var=True)
print_result(ind_result, "Drug vs Placebo BP reduction")
plot_t_distribution(ind_result['statistic'], df=len(drug_bp)+len(placebo_bp)-2,
                    title='Clinical Trial: Drug vs Placebo (Pooled T)',
                    save_path=f'{RESULTS_DIR}/04_clinical_pooled_t.png')

# Welch's T-test
print("\n   ── Test 2b: Welch's T-test (does not assume equal variances)")
welch_result = welch_t(drug_bp, placebo_bp, alpha=0.05)
print_result(welch_result, "Drug vs Placebo BP reduction (Welch)")
plot_ab_test(clinical, 'bp_reduction', 'group',
             save_path=f'{RESULTS_DIR}/05_clinical_ab_plot.png')

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: E-commerce A/B Test — Z-test (proportions) & T-test (revenue)
# "Did the new UI increase conversion rate and revenue per user?"
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  DATASET 3: E-commerce A/B Test")
print("   File: data/ab_test.csv")
print("   Goal: Does the new UI (treatment) improve conversion & revenue?\n")

ab = pd.read_csv(f'{DATA_DIR}/ab_test.csv')
ctrl = ab[ab['variant']=='control']
trt  = ab[ab['variant']=='treatment']

n1, x1 = len(ctrl), ctrl['converted'].sum()
n2, x2 = len(trt),  trt['converted'].sum()
print(f"   Control:   n={n1}, conversions={x1}, rate={x1/n1:.4f} ({x1/n1*100:.2f}%)")
print(f"   Treatment: n={n2}, conversions={x2}, rate={x2/n2:.4f} ({x2/n2*100:.2f}%)")

# Z-test for proportions (conversion rate)
print("\n   ── Test 3a: Two-sample Z-test for proportions (conversion rate)")
prop_result = two_sample_z_proportions(n1, x1, n2, x2, alpha=0.05)
print_result(prop_result, "Conversion rate: Control vs Treatment")
plot_z_distribution(prop_result['statistic'], title='A/B Test: Conversion Rate Z-test',
                    save_path=f'{RESULTS_DIR}/06_ab_conversion_z.png')

# T-test for session duration
print("\n   ── Test 3b: Independent T-test — session duration")
sess_result = independent_t(ctrl['session_duration_sec'], trt['session_duration_sec'])
print_result(sess_result, "Session duration: Control vs Treatment")
plot_ab_test(ab, 'session_duration_sec', 'variant',
             save_path=f'{RESULTS_DIR}/07_ab_session_duration.png')

# T-test for pages viewed
print("\n   ── Test 3c: Independent T-test — pages viewed")
pages_result = independent_t(ctrl['pages_viewed'], trt['pages_viewed'])
print_result(pages_result, "Pages viewed: Control vs Treatment")

# Revenue among converters only
ctrl_rev = ctrl[ctrl['converted']==1]['revenue']
trt_rev  = trt[ trt['converted']==1]['revenue']
print(f"\n   Revenue (converters only): ctrl mean=${ctrl_rev.mean():.2f}, trt mean=${trt_rev.mean():.2f}")
print("\n   ── Test 3d: Welch T-test — revenue per converter")
rev_result = welch_t(ctrl_rev, trt_rev)
print_result(rev_result, "Revenue per converter: Welch T-test")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4: Customer Survey — Chi-square independence & Goodness-of-fit
# "Is satisfaction independent of gender? Are purchase patterns uniform?"
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  DATASET 4: Customer Survey — Chi-square Tests")
print("   File: data/customer_survey.csv")
print("   Goal: Test independence between categorical variables\n")

survey = pd.read_csv(f'{DATA_DIR}/customer_survey.csv')

# Test 4a: Gender × Satisfaction
print("   ── Test 4a: Gender vs Satisfaction (independence)")
chi_result = chi_square_independence(survey, 'gender', 'satisfaction')
print_result(chi_result, "Gender × Satisfaction")
plot_contingency_heatmap(survey, 'gender', 'satisfaction',
                         save_path=f'{RESULTS_DIR}/08_gender_satisfaction_chi.png')
plot_chi_distribution(chi_result['statistic'], df=chi_result['df'],
                      title='Gender × Satisfaction Independence Test',
                      save_path=f'{RESULTS_DIR}/09_chi_distribution.png')

# Test 4b: Region × Recommendation
print("\n   ── Test 4b: Region vs Recommendation (independence)")
chi_region = chi_square_independence(survey, 'region', 'recommend')
print_result(chi_region, "Region × Recommendation")

# Test 4c: Product category × Purchase frequency
print("\n   ── Test 4c: Product Category vs Purchase Frequency")
chi_prod = chi_square_independence(survey, 'product_category', 'purchase_frequency')
print_result(chi_prod, "Product Category × Purchase Frequency")
plot_contingency_heatmap(survey, 'product_category', 'purchase_frequency',
                         save_path=f'{RESULTS_DIR}/10_product_frequency_chi.png')

# Test 4d: Goodness-of-fit — Are purchase frequencies uniformly distributed?
print("\n   ── Test 4d: Goodness-of-fit — Purchase frequency distribution")
freq_counts = survey['purchase_frequency'].value_counts().sort_index()
categories = freq_counts.index.tolist()
observed   = freq_counts.values
expected_probs = [0.25] * len(categories)   # H₀: uniform distribution
gof_result = chi_square_goodness_of_fit(observed, expected_probs, categories)
print_result(gof_result, "Purchase frequency — uniform distribution?")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5: Employee Training — Paired T-test
# "Did the training program significantly improve test scores?"
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  DATASET 5: Employee Training — Paired T-test")
print("   File: data/training_scores.csv")
print("   Goal: Did training significantly improve employee test scores?\n")

training = pd.read_csv(f'{DATA_DIR}/training_scores.csv')
before = training['score_before']
after  = training['score_after']
diff   = training['improvement']
print(f"   n pairs = {len(diff)}")
print(f"   Mean before: {before.mean():.2f}  |  Mean after: {after.mean():.2f}")
print(f"   Mean improvement: {diff.mean():.2f}  |  Std of diff: {diff.std():.2f}")

# Normality of differences (assumption for paired T)
norm_diff = shapiro_normality(diff, 'improvement')
print(f"\n   Normality of differences: p={norm_diff['p_value']} — {norm_diff['conclusion']}")

# Paired T-test
print("\n   ── Test 5a: Paired T-test (right-tailed: did scores IMPROVE?)")
paired_result = paired_t(before, after, alpha=0.05, tail='right')
print_result(paired_result, "Training: Before vs After scores (paired)")
plot_paired_test(before.values, after.values,
                 save_path=f'{RESULTS_DIR}/11_training_paired_t.png')
plot_t_distribution(paired_result['statistic'], df=len(diff)-1,
                    tail='right', title='Training Program: Paired T-test',
                    save_path=f'{RESULTS_DIR}/12_training_t_distribution.png')

# By department
print("\n   ── Test 5b: Paired T-test by department")
for dept in training['department'].unique():
    sub = training[training['department']==dept]
    if len(sub) >= 5:
        r = paired_t(sub['score_before'], sub['score_after'], tail='right')
        sig = '✗ SIGNIFICANT' if r['reject_H0'] else '✓ not significant'
        print(f"   {dept:15s}: n={len(sub):2d}, mean_improvement={sub['improvement'].mean():+.2f}, "
              f"p={r['p_value']}, d={r['effect_size_d']} — {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Distribution shape comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n▶  BONUS: Distribution shape comparison plots")
plot_distribution_shapes(save_path=f'{RESULTS_DIR}/00_distribution_shapes.png')
print("   Saved: results/00_distribution_shapes.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "═"*65)
print("  SUMMARY OF ALL HYPOTHESIS TESTS")
print("═"*65)
summary = [
    ("Manufacturing", "One-sample Z-test", z_result['statistic'],      z_result['p_value'],      z_result['reject_H0']),
    ("Manufacturing", "One-sample T-test", t_result['statistic'],      t_result['p_value'],      t_result['reject_H0']),
    ("Clinical",      "Independent T (pooled)", ind_result['statistic'], ind_result['p_value'],  ind_result['reject_H0']),
    ("Clinical",      "Welch T-test",      welch_result['statistic'],  welch_result['p_value'],  welch_result['reject_H0']),
    ("A/B Test",      "Z-test proportions",prop_result['statistic'],   prop_result['p_value'],   prop_result['reject_H0']),
    ("A/B Test",      "T-test session dur",sess_result['statistic'],   sess_result['p_value'],   sess_result['reject_H0']),
    ("A/B Test",      "T-test pages viewed",pages_result['statistic'], pages_result['p_value'],  pages_result['reject_H0']),
    ("Survey",        "Chi-sq Gender×Sat", chi_result['statistic'],    chi_result['p_value'],    chi_result['reject_H0']),
    ("Survey",        "Chi-sq Region×Rec", chi_region['statistic'],   chi_region['p_value'],    chi_region['reject_H0']),
    ("Survey",        "Chi-sq GOF uniform",gof_result['statistic'],   gof_result['p_value'],    gof_result['reject_H0']),
    ("Training",      "Paired T-test",     paired_result['statistic'],paired_result['p_value'], paired_result['reject_H0']),
]
print(f"\n  {'Dataset':<16} {'Test':<25} {'Statistic':>10} {'p-value':>10} {'Reject H₀'}")
print("  " + "─"*63)
for ds, test, stat, pval, reject in summary:
    icon = "✗ YES" if reject else "✓ NO "
    print(f"  {ds:<16} {test:<25} {str(stat):>10} {str(pval):>10}   {icon}")

print("\n  All plots saved to: results/")
print("═"*65 + "\n")
