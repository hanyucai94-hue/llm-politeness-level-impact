#!/usr/bin/env python3
"""
Statistical analysis to test if politeness level significantly affects model accuracy.

This script performs:
1. Paired t-tests between different politeness levels
2. ANOVA to test overall effect of politeness
3. Effect size calculations (Cohen's d)
4. Visualization of results
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

def cohens_d(group1, group2):
    """Calculate Cohen's d for effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_politeness_effect(csv_file):
    """
    Analyze the effect of politeness on model accuracy.
    
    Args:
        csv_file: Path to per_question_accuracy_*.csv file
    
    Returns:
        Dictionary with paths to all generated output files
    """
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: {csv_file}")
    print(f"{'='*80}\n")
    
    # Prepare output directory and file paths
    output_dir = os.path.dirname(csv_file)
    filename = os.path.basename(csv_file)
    # Extract domain suffix from filename (e.g., "per_question_accuracy_humanities.csv" -> "_humanities")
    base_name = filename.replace('per_question_accuracy', '').replace('.csv', '')
    
    output_files = {
        'summary_stats': os.path.join(output_dir, f'stats_summary{base_name}.csv'),
        'anova_results': os.path.join(output_dir, f'stats_anova{base_name}.csv'),
        'pairwise_tests': os.path.join(output_dir, f'stats_pairwise{base_name}.csv'),
        'effect_sizes': os.path.join(output_dir, f'stats_effect_sizes{base_name}.csv'),
        'practical_significance': os.path.join(output_dir, f'stats_practical{base_name}.csv'),
        'visualization': os.path.join(output_dir, f'stats_visualization{base_name}.png')
    }
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Pivot to get one row per question, columns for each tone
    pivot_df = df.pivot(index='QID', columns='Tone', values='Average Accuracy (%)')
    
    # Summary statistics
    summary_stats_data = []
    for tone in pivot_df.columns:
        mean_acc = pivot_df[tone].mean()
        std_acc = pivot_df[tone].std()
        min_acc = pivot_df[tone].min()
        max_acc = pivot_df[tone].max()
        median_acc = pivot_df[tone].median()
        
        summary_stats_data.append({
            'Tone': tone,
            'Mean (%)': round(mean_acc, 2),
            'Std Dev (%)': round(std_acc, 2),
            'Min (%)': round(min_acc, 2),
            'Max (%)': round(max_acc, 2),
            'Median (%)': round(median_acc, 2),
            'N Questions': len(pivot_df[tone])
        })
    
    summary_stats_df = pd.DataFrame(summary_stats_data)
    summary_stats_df.to_csv(output_files['summary_stats'], index=False)
    
    # ===== 1. REPEATED MEASURES ANOVA =====
    # Prepare data for repeated measures ANOVA
    # Each question is measured under all 5 conditions (tones)
    f_stat, p_value = stats.f_oneway(*[pivot_df[tone] for tone in pivot_df.columns])
    
    interpretation = "Politeness level DOES significantly affect accuracy!" if p_value < 0.05 else "No significant effect of politeness level on accuracy."
    
    # Save ANOVA results
    anova_df = pd.DataFrame([{
        'Test': 'One-Way ANOVA',
        'F-statistic': round(f_stat, 4),
        'p-value': f'{p_value:.6f}',
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No',
        'Significance Level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns',
        'Interpretation': interpretation
    }])
    anova_df.to_csv(output_files['anova_results'], index=False)
    
    # ===== 2. PAIRWISE PAIRED T-TESTS =====
    tones = list(pivot_df.columns)
    results = []
    
    for tone1, tone2 in combinations(tones, 2):
        # Paired t-test (same questions, different conditions)
        t_stat, p_val = stats.ttest_rel(pivot_df[tone1], pivot_df[tone2])
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(pivot_df[tone1], pivot_df[tone2])
        
        # Mean difference
        mean_diff = pivot_df[tone1].mean() - pivot_df[tone2].mean()
        
        results.append({
            'Comparison': f'{tone1} vs {tone2}',
            'Mean Diff': mean_diff,
            't-statistic': t_stat,
            'p-value': p_val,
            'Cohen\'s d': effect_size,
            'Significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')
    
    # Bonferroni correction for multiple comparisons
    n_comparisons = len(results)
    bonferroni_alpha = 0.05 / n_comparisons
    results_df['Bonferroni Significant'] = results_df['p-value'] < bonferroni_alpha
    results_df['Bonferroni α'] = bonferroni_alpha
    
    # Save pairwise test results
    results_df.to_csv(output_files['pairwise_tests'], index=False)
    
    # ===== 3. EFFECT SIZE INTERPRETATION =====
    effect_size_data = []
    for _, row in results_df.iterrows():
        cohens_d_value = row["Cohen's d"]
        d = abs(cohens_d_value)
        if d < 0.2:
            size = "Negligible"
        elif d < 0.5:
            size = "Small"
        elif d < 0.8:
            size = "Medium"
        else:
            size = "Large"
        
        effect_size_data.append({
            'Comparison': row['Comparison'],
            "Cohen's d": cohens_d_value,
            'Absolute |d|': d,
            'Effect Size': size
        })
    
    effect_size_df = pd.DataFrame(effect_size_data)
    effect_size_df.to_csv(output_files['effect_sizes'], index=False)
    
    # ===== 4. PRACTICAL SIGNIFICANCE =====
    max_tone = pivot_df.mean().idxmax()
    min_tone = pivot_df.mean().idxmin()
    max_acc = pivot_df.mean().max()
    min_acc = pivot_df.mean().min()
    diff = max_acc - min_acc
    
    if diff < 0.5:
        practical_interp = "Negligible practical difference (< 0.5 pp)"
    elif diff < 1.0:
        practical_interp = "Small practical difference (< 1 pp)"
    elif diff < 2.0:
        practical_interp = "Moderate practical difference (< 2 pp)"
    else:
        practical_interp = "Large practical difference (≥ 2 pp)"
    
    # Save practical significance
    practical_df = pd.DataFrame([{
        'Best Tone': max_tone,
        'Best Accuracy (%)': round(max_acc, 2),
        'Worst Tone': min_tone,
        'Worst Accuracy (%)': round(min_acc, 2),
        'Absolute Difference (pp)': round(diff, 2),
        'Relative Difference (%)': round((diff/min_acc)*100, 2),
        'Interpretation': practical_interp
    }])
    practical_df.to_csv(output_files['practical_significance'], index=False)
    
    # ===== 5. VISUALIZATION =====
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Politeness Effect Analysis\n{csv_file}', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plot
    ax1 = axes[0, 0]
    pivot_df.boxplot(ax=ax1)
    ax1.set_title('Distribution of Accuracy by Politeness Level')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Politeness Level')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar plot with error bars
    ax2 = axes[0, 1]
    means = pivot_df.mean()
    stds = pivot_df.std()
    x_pos = np.arange(len(means))
    ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(means.index, rotation=45, ha='right')
    ax2.set_title('Mean Accuracy by Politeness Level (± SD)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Heatmap of p-values
    ax3 = axes[1, 0]
    p_matrix = np.ones((len(tones), len(tones)))
    for i, tone1 in enumerate(tones):
        for j, tone2 in enumerate(tones):
            if i != j:
                _, p_val = stats.ttest_rel(pivot_df[tone1], pivot_df[tone2])
                p_matrix[i, j] = p_val
    
    sns.heatmap(p_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                xticklabels=tones, yticklabels=tones, ax=ax3,
                cbar_kws={'label': 'p-value'}, vmin=0, vmax=0.1)
    ax3.set_title('Pairwise T-Test P-Values')
    
    # Plot 4: Effect sizes
    ax4 = axes[1, 1]
    effect_sizes = []
    labels = []
    for tone1, tone2 in combinations(tones, 2):
        d = cohens_d(pivot_df[tone1], pivot_df[tone2])
        effect_sizes.append(abs(d))
        labels.append(f'{tone1}\nvs\n{tone2}')
    
    colors = ['red' if d >= 0.8 else 'orange' if d >= 0.5 else 'yellow' if d >= 0.2 else 'green' 
              for d in effect_sizes]
    ax4.barh(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('|Cohen\'s d|')
    ax4.set_title('Effect Sizes (Cohen\'s d)')
    ax4.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
    ax4.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_files['visualization'], dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== SUMMARY =====
    if p_value < 0.05 and diff >= 1.0:
        recommendation = "Politeness DOES matter - both statistically and practically significant!"
    elif p_value < 0.05:
        recommendation = "Statistically significant but small practical effect."
    elif diff >= 1.0:
        recommendation = "Large practical difference but not statistically significant (may need more data)."
    else:
        recommendation = "No significant effect of politeness on accuracy."
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Statistical Significance: {'YES' if p_value < 0.05 else 'NO'} (p = {p_value:.6f})")
    print(f"Best Tone: {max_tone} ({max_acc:.2f}%)")
    print(f"Worst Tone: {min_tone} ({min_acc:.2f}%)")
    print(f"Difference: {diff:.2f} pp")
    print(f"Conclusion: {recommendation}")
    print(f"{'='*80}")
    print(f"\n✅ All results saved to: {output_dir}/")
    print(f"   • stats_summary{base_name}.csv")
    print(f"   • stats_anova{base_name}.csv")
    print(f"   • stats_pairwise{base_name}.csv")
    print(f"   • stats_effect_sizes{base_name}.csv")
    print(f"   • stats_practical{base_name}.csv")
    print(f"   • stats_visualization{base_name}.png\n")
    
    return output_files

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default file
        csv_file = 'results-gpt-4o-mini/per_question_accuracy_stem.csv'
    
    analyze_politeness_effect(csv_file)

