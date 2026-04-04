"""
Create visualizations for the research paper.
- Performance comparison bar chart
- Results table
- Summary statistics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent.parent / "results" / "metrics" / "results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

plots_dir = Path(__file__).parent.parent / "results" / "plots"
plots_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ============================================================================
# 1. Performance Comparison Bar Chart
# ============================================================================
models = ['GCN', 'GraphSAGE', 'GAT', 'R-GCN']
test_auc = [results[m]['test_metrics']['auc'] for m in models]
test_ap = [results[m]['test_metrics']['ap'] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# AUC comparison
x = np.arange(len(models))
bars1 = ax1.bar(x, test_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test AUC-ROC', fontsize=12, fontweight='bold')
ax1.set_title('Link Prediction Performance (AUC)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim([0.7, 1.0])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# AP comparison
bars2 = ax2.bar(x, test_ap, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Average Precision', fontsize=12, fontweight='bold')
ax2.set_title('Link Prediction Performance (AP)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim([0.7, 1.0])
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {plots_dir / 'performance_comparison.png'}")

# ============================================================================
# 2. Training Time Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

times = [results[m]['time'] for m in models]
bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.set_yscale('log')  # Log scale since GCN is much slower
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, time in zip(bars, times):
    height = bar.get_height()
    if time < 1:
        label = f'{time:.1f}s'
    elif time < 60:
        label = f'{time:.1f}s'
    else:
        label = f'{time/60:.1f}m'
    ax.text(bar.get_x() + bar.get_width()/2., height,
            label,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'training_time.png', dpi=300, bbox_inches='tight')
print(f"Saved: {plots_dir / 'training_time.png'}")

# ============================================================================
# 3. Performance vs Efficiency Scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for i, model in enumerate(models):
    auc = results[model]['test_metrics']['auc']
    time = results[model]['time']
    ax.scatter(time, auc, s=500, c=[colors[i]], alpha=0.6,
               edgecolors='black', linewidth=2, label=model)
    ax.text(time, auc, model, ha='center', va='center',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
ax.set_title('Performance vs Training Efficiency', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='lower right')

# Add quadrant labels
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
ax.text(0.9, 0.96, 'High Performance\nFast Training',
        ha='left', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(plots_dir / 'performance_vs_time.png', dpi=300, bbox_inches='tight')
print(f"Saved: {plots_dir / 'performance_vs_time.png'}")

# ============================================================================
# 4. Create LaTeX table
# ============================================================================
latex_table = r"""\begin{table}[h]
\centering
\caption{Performance Comparison of Graph Neural Network Models on Zero-Shot Antibiotic Resistance Prediction}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Val AUC} & \textbf{Test AUC} & \textbf{Test AP} & \textbf{Time (s)} \\
\midrule
"""

for model in models:
    r = results[model]
    val_auc = r['val_auc']
    test_auc = r['test_metrics']['auc']
    test_ap = r['test_metrics']['ap']
    time = r['time']

    # Bold the best values
    if test_auc == max(test_auc for r in results.values() if 'test_metrics' in r):
        test_auc_str = f"\\textbf{{{test_auc:.4f}}}"
    else:
        test_auc_str = f"{test_auc:.4f}"

    latex_table += f"{model} & {val_auc:.4f} & {test_auc_str} & {test_ap:.4f} & {time:.1f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

tables_dir = Path(__file__).parent.parent / "results" / "tables"
tables_dir.mkdir(exist_ok=True)

with open(tables_dir / 'results_table.tex', 'w') as f:
    f.write(latex_table)

print(f"Saved: {tables_dir / 'results_table.tex'}")

# ============================================================================
# 5. Summary statistics
# ============================================================================
summary = {
    'best_model': max(results.keys(), key=lambda k: results[k]['test_metrics']['auc'] if 'test_metrics' in results[k] else 0),
    'best_auc': max(r['test_metrics']['auc'] for r in results.values() if 'test_metrics' in r),
    'fastest_model': min(results.keys(), key=lambda k: results[k]['time'] if 'time' in results[k] else float('inf')),
    'fastest_time': min(r['time'] for r in results.values() if 'time' in r),
}

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Best Model: {summary['best_model']} (AUC: {summary['best_auc']:.4f})")
print(f"Fastest Model: {summary['fastest_model']} (Time: {summary['fastest_time']:.2f}s)")
print("="*60)

print("\nAll visualizations created successfully!")
