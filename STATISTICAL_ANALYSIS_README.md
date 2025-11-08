# Statistical Analysis Script - User Guide

## Overview

The `statistical_analysis.py` script performs comprehensive statistical analysis to determine if politeness level significantly affects model accuracy.

## Usage

```bash
python statistical_analysis.py <path_to_per_question_accuracy_file.csv>
```

### Example:

```bash
python statistical_analysis.py results-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8/per_question_accuracy_humanities.csv
```

## Output Files

The script generates 5 CSV files and 1 visualization:

### 1. `stats_summary_<domain>.csv`

**Summary statistics for each politeness level**

Columns:

- `Tone`: Politeness level (e.g., Very Friendly, Neutral, Rude)
- `Mean (%)`: Average accuracy across all questions
- `Std Dev (%)`: Standard deviation
- `Min (%)`: Minimum accuracy
- `Max (%)`: Maximum accuracy
- `Median (%)`: Median accuracy
- `N Questions`: Number of questions tested

**Use for**: Quick overview of performance by tone

---

### 2. `stats_anova_<domain>.csv`

**One-Way ANOVA results**

Columns:

- `Test`: Type of test (One-Way ANOVA)
- `F-statistic`: F-statistic value
- `p-value`: Statistical significance
- `Significant (α=0.05)`: Yes/No
- `Significance Level`: **\*, **, \*, or ns
- `Interpretation`: Plain English interpretation

**Use for**: Determining if politeness has ANY effect on accuracy

**Interpretation**:

- p < 0.05: Politeness level DOES affect accuracy
- p ≥ 0.05: No significant effect

---

### 3. `stats_pairwise_<domain>.csv`

**Pairwise comparisons between all tone pairs**

Columns:

- `Comparison`: Which two tones are being compared
- `Mean Diff`: Difference in mean accuracy (percentage points)
- `t-statistic`: t-test statistic
- `p-value`: Statistical significance
- `Cohen's d`: Effect size
- `Significant`: **\*, **, \*, or ns
- `Bonferroni Significant`: True/False (after correction for multiple comparisons)
- `Bonferroni α`: Corrected significance threshold

**Use for**: Identifying which specific tone pairs differ significantly

**Interpretation**:

- p < 0.05: These two tones produce significantly different accuracies
- Look at `Bonferroni Significant` for more conservative results

---

### 4. `stats_effect_sizes_<domain>.csv`

**Effect sizes for all comparisons**

Columns:

- `Comparison`: Which two tones are being compared
- `Cohen's d`: Effect size (can be positive or negative)
- `Absolute |d|`: Absolute value of effect size
- `Effect Size`: Interpretation (Negligible, Small, Medium, Large)

**Use for**: Understanding the magnitude of differences

**Interpretation**:

- |d| < 0.2: Negligible effect
- |d| < 0.5: Small effect
- |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

---

### 5. `stats_practical_<domain>.csv`

**Practical significance summary**

Columns:

- `Best Tone`: Tone with highest accuracy
- `Best Accuracy (%)`: Accuracy of best tone
- `Worst Tone`: Tone with lowest accuracy
- `Worst Accuracy (%)`: Accuracy of worst tone
- `Absolute Difference (pp)`: Difference in percentage points
- `Relative Difference (%)`: Relative percentage difference
- `Interpretation`: Practical significance level

**Use for**: Understanding real-world impact

**Interpretation**:

- < 0.5 pp: Negligible practical difference
- < 1 pp: Small practical difference
- < 2 pp: Moderate practical difference
- ≥ 2 pp: Large practical difference

---

### 6. `stats_visualization_<domain>.png`

**Visual summary with 4 plots**

1. **Box plot**: Distribution of accuracy by tone
2. **Bar chart**: Mean accuracy with error bars
3. **Heatmap**: P-values for all pairwise comparisons
4. **Effect sizes**: Cohen's d for all comparisons

**Use for**: Presentations, papers, quick visual inspection

---

## Statistical Methods

### 1. Repeated Measures Design

- Same questions tested under different politeness conditions
- Uses **paired t-tests** (more powerful than independent t-tests)

### 2. Multiple Comparison Correction

- **Bonferroni correction** applied to control family-wise error rate
- With 5 tones, there are 10 pairwise comparisons
- Corrected α = 0.05 / 10 = 0.005

### 3. Effect Size

- **Cohen's d** measures standardized difference between groups
- Helps distinguish statistical vs practical significance

---

## Interpreting Results

### Scenario 1: Statistically Significant + Large Effect

**Example**: p < 0.001, difference = 2.5 pp, Cohen's d = 0.9

**Interpretation**: Politeness DOES matter! Both statistically and practically significant.

**Action**: Report this as a key finding. Politeness level affects model performance.

---

### Scenario 2: Statistically Significant + Small Effect

**Example**: p = 0.02, difference = 0.3 pp, Cohen's d = 0.15

**Interpretation**: Statistically significant but negligible practical effect.

**Action**: Mention in paper but note the effect is too small to matter in practice.

---

### Scenario 3: Not Significant + Large Effect

**Example**: p = 0.08, difference = 1.8 pp, Cohen's d = 0.7

**Interpretation**: Large practical difference but not statistically significant.

**Action**: May need more data (more questions or more runs). Consider as exploratory finding.

---

### Scenario 4: Not Significant + Small Effect

**Example**: p = 0.45, difference = 0.2 pp, Cohen's d = 0.08

**Interpretation**: No evidence that politeness affects accuracy.

**Action**: Report null finding. Politeness level doesn't matter for this model.

---

## Common Questions

### Q: What if ANOVA is significant but no pairwise tests are?

**A**: This can happen due to multiple comparison correction. Look at uncorrected p-values to see which pairs are close to significant.

### Q: What if results differ between domains (humanities vs stem)?

**A**: This is interesting! It suggests the effect of politeness may be domain-specific. Report both and discuss the difference.

### Q: How many runs do I need?

**A**: You have 3 runs per question, which is good. More runs would increase power but may not be necessary if effects are clear.

### Q: Should I use one-tailed or two-tailed tests?

**A**: The script uses two-tailed tests (more conservative). Use one-tailed only if you have a strong directional hypothesis.

---

## For Your Paper

### Key Results to Report:

1. **ANOVA result**: "A one-way repeated measures ANOVA revealed [significant/no significant] effect of politeness level on accuracy, F(4, 1996) = X.XX, p = X.XXX"

2. **Best vs Worst**: "The [tone] condition yielded the highest accuracy (X.XX%), while [tone] yielded the lowest (X.XX%), a difference of X.XX percentage points"

3. **Effect size**: "The effect size was [negligible/small/medium/large] (Cohen's d = X.XX)"

4. **Pairwise comparisons**: "Post-hoc pairwise comparisons with Bonferroni correction revealed significant differences between [list pairs]"

5. **Practical significance**: "Despite statistical significance, the practical impact was [negligible/small/moderate/large], suggesting that politeness level [does/does not] meaningfully affect model performance"

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solution**: Install required packages:

```bash
pip install pandas numpy scipy matplotlib seaborn
```

### Error: "ValueError: Index contains duplicate entries"

**Solution**: Make sure you're using the `per_question_accuracy_*.csv` file, not `run_summary_results_*.csv`

### Visualization doesn't show

**Solution**: The PNG file is saved automatically. Check the output directory.

---

## Contact

For questions or issues, check the script comments or modify the script to suit your specific needs.

