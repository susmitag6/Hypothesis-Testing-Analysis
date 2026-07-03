# Statistical Hypothesis Testing Portfolio

> A complete, end-to-end hypothesis testing project covering **Z-tests**, **T-tests**, and **Chi-square tests** across 5 real-world datasets — with assumption checking, effect sizes, visualizations, and a reusable Python library.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Datasets](#datasets)
4. [Tests Implemented](#tests-implemented)
5. [Key Results](#key-results)
6. [Statistical Concepts Covered](#statistical-concepts-covered)
7. [How to Run](#how-to-run)
8. [Dependencies](#dependencies)
9. [Visualizations](#visualizations)
10. [What I Learned](#what-i-learned)

---

## Project Overview

This portfolio demonstrates a rigorous, practical approach to hypothesis testing — the kind used daily by data scientists in industry. Each analysis follows the full statistical workflow:

```
State hypotheses → Check assumptions → Run test →
Interpret statistic + p-value → Report effect size → Draw conclusion
```

**Why this matters:** Statistical significance alone is not enough. This project consistently reports **effect sizes** (Cohen's d, Cramer's V), **confidence intervals**, and **practical significance** alongside p-values — reflecting production-grade statistical practice.

---

## Repository Structure

```
hypothesis_testing/
│
├── data/
│   ├── generate_datasets.py     # Reproducible dataset generation (seed=42)
│   ├── clinical_trial.csv       # 300 patients: drug vs placebo BP reduction
│   ├── ab_test.csv              # 5,000 users: e-commerce UI A/B test
│   ├── manufacturing.csv        # 200 batches: product weight QC
│   ├── customer_survey.csv      # 800 respondents: satisfaction survey
│   └── training_scores.csv      # 80 employees: before/after training
│
├── src/
│   ├── hypothesis_tests.py      # Core testing library (Z, T, Chi-square)
│   └── visualizations.py        # Publication-quality plotting functions
│
├── notebooks/project_report.ipynb
│
├── results/                     # All generated plots (PNG)
│   ├── 00_distribution_shapes.png
│   ├── 01_manufacturing_z_test.png
│   ├── 02_manufacturing_t_test.png
│   ├── 03_clinical_assumption_checks.png
│   ├── 04_clinical_pooled_t.png
│   ├── 05_clinical_ab_plot.png
│   ├── 06_ab_conversion_z.png
│   ├── 07_ab_session_duration.png
│   ├── 08_gender_satisfaction_chi.png
│   ├── 09_chi_distribution.png
│   ├── 10_product_frequency_chi.png
│   ├── 11_training_paired_t.png
│   └── 12_training_t_distribution.png
│
├── run_analysis.py              # Master script — runs everything end-to-end
└── README.md
```

---

## Datasets

| Dataset | File | n | Business Question |
|---------|------|---|-------------------|
| Clinical Trial | `clinical_trial.csv` | 300 | Does the drug lower blood pressure vs. placebo? |
| E-commerce A/B | `ab_test.csv` | 5,000 | Does the new UI increase conversion rate? |
| Manufacturing QC | `manufacturing.csv` | 200 | Are products meeting the 500g weight specification? |
| Customer Survey | `customer_survey.csv` | 800 | Is satisfaction independent of gender? |
| Employee Training | `training_scores.csv` | 80 | Did training significantly improve test scores? |

All datasets are synthetically generated with `numpy.random.seed(42)` for full reproducibility, but are designed to mimic realistic distributions and effect sizes found in industry.

---

## Tests Implemented

### Z-Tests (`src/hypothesis_tests.py`)

#### `one_sample_z(sample, mu0, sigma, alpha, tail)`
Tests whether a sample mean equals a hypothesized population mean when **σ is known**.

```python
from src.hypothesis_tests import one_sample_z

result = one_sample_z(weights, mu0=500, sigma=3.5, alpha=0.05, tail='two')
# Returns: statistic, p-value, CI, effect size, decision
```

**Applied to:** Manufacturing — testing if product weight = 500g  
**Result:** Z = 2.95, p = 0.003 → **Reject H₀** — machine is miscalibrated

---

#### `two_sample_z_proportions(n1, x1, n2, x2, alpha, tail)`
Tests whether two conversion rates (proportions) are equal.

```python
result = two_sample_z_proportions(n1=2500, x1=102, n2=2500, x2=132, alpha=0.05)
```

**Applied to:** A/B test — control (4.08%) vs treatment (5.28%) conversion  
**Result:** Z = −2.01, p = 0.045 → **Reject H₀** — treatment converts significantly better

---

### T-Tests

#### `one_sample_t(sample, mu0, alpha, tail)`
Tests a sample mean against a known value when **σ is unknown** (the more realistic case).

```python
result = one_sample_t(weights, mu0=500, alpha=0.05)
```

**Applied to:** Manufacturing (σ estimated from data)  
**Result:** t = 3.23, p = 0.001 → **Reject H₀**, Cohen's d = 0.23 (small effect)

---

#### `independent_t(group1, group2, alpha, tail, equal_var=None)`
Compares means of two independent groups. Automatically runs **Levene's test** to select pooled vs. Welch variant.

```python
result = independent_t(drug_bp, placebo_bp, alpha=0.05)
# Auto-detects equal_var via Levene's test
```

**Applied to:** Clinical trial — drug vs placebo BP reduction  
**Result:** t = −5.87, p < 0.001, Cohen's d = 0.68 (medium effect) → **Drug is effective**

---

#### `welch_t(group1, group2, alpha, tail)`
Welch's T-test — does **not** assume equal variances. Preferred when Levene's test is significant.

```python
result = welch_t(ctrl_revenue, trt_revenue)
```

**Applied to:** A/B test — revenue per converter  
**Result:** t = 0.016, p = 0.987 → **Fail to reject H₀** — revenue unchanged

---

#### `paired_t(before, after, alpha, tail)`
Tests the mean of paired differences. Far more powerful than independent T when the same subjects are measured twice.

```python
result = paired_t(scores_before, scores_after, tail='right')
```

**Applied to:** Employee training program  
**Result:** t = 14.49, p < 10⁻²⁴, Cohen's d = 1.62 (large effect) → **Training works**

---

### Chi-Square Tests

#### `chi_square_independence(data, col1, col2, alpha)`
Tests whether two categorical variables are associated. Reports **Cramer's V** as effect size and checks all expected count assumptions.

```python
result = chi_square_independence(survey, 'gender', 'satisfaction')
# Checks: all expected counts ≥ 5; reports Cramer's V
```

**Applied to:** Gender × Satisfaction, Region × Recommendation, Product × Purchase frequency  
**Result (Gender×Sat):** χ² = 16.87, p = 0.032, V = 0.10 (small effect) → **Associated**

---

#### `chi_square_goodness_of_fit(observed, expected_probs, categories, alpha)`
Tests whether observed counts match a theoretical distribution.

```python
result = chi_square_goodness_of_fit(
    observed=[40, 200, 360, 200],
    expected_probs=[0.25, 0.25, 0.25, 0.25],
    categories=['Daily','Weekly','Monthly','Rarely']
)
```

**Applied to:** Purchase frequency — is it uniformly distributed?  
**Result:** χ² = 241.87, p ≈ 10⁻⁵² → **Reject H₀** — strongly non-uniform

---

## Key Results

| Dataset | Test | Statistic | p-value | Decision | Effect Size |
|---------|------|-----------|---------|----------|-------------|
| Manufacturing | One-sample Z | 2.95 | 0.0032 | **Reject H₀** | d = 0.23 (small) |
| Manufacturing | One-sample T | 3.23 | 0.0014 | **Reject H₀** | d = 0.23 (small) |
| Clinical | Pooled T | −5.87 | <0.001 | **Reject H₀** | d = 0.68 (medium) |
| Clinical | Welch T | −5.87 | <0.001 | **Reject H₀** | d = 0.68 (medium) |
| A/B Test | Z (proportions) | −2.01 | 0.0446 | **Reject H₀** | Lift = +29% |
| A/B Test | T (sessions) | −5.44 | <0.001 | **Reject H₀** | d = 0.15 (negligible) |
| A/B Test | T (pages) | −6.68 | <0.001 | **Reject H₀** | d = 0.19 (negligible) |
| A/B Test | Welch T (revenue) | 0.016 | 0.987 | Fail to reject | d ≈ 0 |
| Survey | Chi-sq (Gender×Sat) | 16.87 | 0.0315 | **Reject H₀** | V = 0.10 (small) |
| Survey | Chi-sq (Region×Rec) | 7.19 | 0.304 | Fail to reject | V = 0.07 (negligible) |
| Survey | Chi-sq GOF | 241.87 | <0.001 | **Reject H₀** | — |
| Training | Paired T | 14.49 | <0.001 | **Reject H₀** | d = 1.62 (large) |

> **Notable insight:** The A/B test shows that while conversion rate and engagement metrics improved significantly, **revenue per converter did not change** (p = 0.987). This is a common real-world finding — more users converting doesn't mean each converter spends more. Statistical significance ≠ business value.

---

## Statistical Concepts Covered

### Hypothesis Testing Framework
- Null (H₀) and alternative (H₁) hypothesis formulation
- One-tailed vs two-tailed tests
- Type I error (α) and Type II error (β)
- p-value interpretation and common misinterpretations

### The Three Tests in Depth

**Z-test**
- Fixed critical values (±1.96 at α=0.05)
- Requires known population σ
- Standard Normal distribution shape
- Two-proportion Z-test for A/B conversion analysis

**T-test**
- Student's t-distribution with heavier tails than normal
- Critical values grow as df decreases
- Pooled vs Welch variant selection via Levene's test
- Paired design for matched data — eliminates between-subject variance

**Chi-square test**
- Right-skewed, non-negative distribution
- Only right-tailed (no two-tailed)
- Expected count assumption (≥ 5 per cell)
- Cramer's V for effect size
- Goodness-of-fit vs independence test distinction

### Assumption Checking (done before every test)
- **Normality:** Shapiro-Wilk test + Q-Q plots
- **Equal variances:** Levene's test
- **Expected counts:** Chi-square cell check
- **Independence:** By design (random sampling)

### Effect Sizes (always reported alongside p-values)
- **Cohen's d** for Z and T tests: negligible (<0.2), small (0.2–0.5), medium (0.5–0.8), large (>0.8)
- **Cramer's V** for Chi-square: thresholds vary by table dimension
- **Relative lift** for proportion tests

### Critical Value Calculation
- Z: `scipy.stats.norm.ppf(1 - α/2)`
- T: `scipy.stats.t.ppf(1 - α/2, df)`
- χ²: `scipy.stats.chi2.ppf(1 - α, df)`

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/hypothesis-testing-portfolio
cd hypothesis-testing-portfolio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate datasets
python3 data/generate_datasets.py

# 4. Run the full analysis
python3 run_analysis.py

# Results printed to console + plots saved to results/
```

### Using the library standalone

```python
from src.hypothesis_tests import independent_t, chi_square_independence, print_result
import pandas as pd

# Your own data
df = pd.read_csv('your_data.csv')

# Run a test
result = independent_t(df[df['group']=='A']['metric'],
                       df[df['group']=='B']['metric'])
print_result(result, title='My A/B Test')
```

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
```

Install with: `pip install numpy pandas scipy matplotlib seaborn statsmodels`

---

## Visualizations

Each test produces a dedicated plot saved to `results/`:

| Plot | What it shows |
|------|--------------|
| `00_distribution_shapes` | T-distribution vs normal at varying df; Chi-square by k |
| `01/02_manufacturing_*` | Z and T distributions with critical region + test statistic |
| `03_clinical_assumption_checks` | Histograms + Q-Q plots for both groups |
| `04_clinical_pooled_t` | T-distribution with rejection region |
| `05_clinical_ab_plot` | Side-by-side histograms, box plots, mean with CI |
| `06_ab_conversion_z` | Z-distribution for proportion test |
| `07_ab_session_duration` | A/B comparison: distribution + box + mean CI |
| `08/10_*_chi` | Observed vs expected heatmaps + standardized residuals |
| `09_chi_distribution` | Chi-square curve with rejection region |
| `11_training_paired_t` | Individual change lines + difference histogram + Q-Q |
| `12_training_t_distribution` | T-distribution one-tailed test |

---

## What I Learned

1. **Assumption checking is not optional.** Skipping Shapiro-Wilk or Levene's test before a T-test is a common mistake in practice. This project builds checking into the workflow.

2. **Welch's T-test should be the default.** Unless you have strong theoretical reason to assume equal variances, Welch's T-test is always safe. The penalty for using it when variances are actually equal is tiny.

3. **Statistical significance ≠ practical significance.** The A/B test showed highly significant differences in sessions and pages viewed (p < 0.001) with negligible effect sizes (d < 0.2). A business decision should not be made on p-values alone.

4. **Paired design is powerful.** The training dataset shows Cohen's d = 1.62 (large) — partly because the paired design removes between-employee variance that would inflate the error term in an independent test.

5. **Chi-square assumptions matter.** One cell in the Gender×Satisfaction table had an expected count < 5, triggering a warning to consider Fisher's exact test or collapsing categories. Ignoring this can inflate the Type I error rate.

6. **Effect size interpretation depends on context.** A Cohen's d of 0.23 (small) in manufacturing quality control is actually very important — a 0.73g mean deviation from 500g target means thousands of non-compliant units daily at scale.

---

*All analysis reproducible with `numpy.random.seed(42)`. Generated with Python 3.11.*
