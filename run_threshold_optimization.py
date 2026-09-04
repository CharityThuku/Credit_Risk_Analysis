"""
Step 7: Threshold optimization + cost-benefit analysis for Easy Ensemble.

Uses the real y_true/y_proba already saved in model_results.pkl by
run_full_analysis.py -- no retraining needed.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle
import pandas as pd

from credit_risk_threshold_optimization import (
    find_optimal_threshold,
    compare_thresholds,
    cost_benefit_analysis,
    plot_threshold_optimization,
)

with open('model_results.pkl', 'rb') as f:
    data = pickle.load(f)

ee = data['results']['Easy Ensemble']
y_true, y_proba = ee['y_true'], ee['y_proba']

print("="*80)
print("STEP 7: THRESHOLD OPTIMIZATION -- Easy Ensemble")
print("="*80)

# 1. Optimal threshold under three metrics
for metric in ['f1', 'balanced_accuracy', 'g_mean']:
    thr, scores, thresholds, best = find_optimal_threshold(y_true, y_proba, metric=metric)
    print(f"\nOptimal threshold ({metric}): {thr:.2f}  (score = {best:.3f})")

# 2. Business-facing threshold comparison
results_df = compare_thresholds(
    y_true, y_proba,
    thresholds_to_test=[0.3, 0.4, 0.5, 0.6, 0.7]
)
results_df.to_csv('threshold_comparison.csv', index=False)
print("\nSaved threshold_comparison.csv")

# 3. Cost-benefit at the F1-optimal threshold vs. default 0.5
f1_thr, _, _, _ = find_optimal_threshold(y_true, y_proba, metric='f1')

print("\n--- Cost-benefit: default threshold (0.5) ---")
default_impact = cost_benefit_analysis(y_true, y_proba, threshold=0.5)
for k, v in default_impact.items():
    print(f"  {k}: {v}")

print(f"\n--- Cost-benefit: F1-optimal threshold ({f1_thr:.2f}) ---")
optimal_impact = cost_benefit_analysis(y_true, y_proba, threshold=f1_thr)
for k, v in optimal_impact.items():
    print(f"  {k}: {v}")

print(f"\nTotal cost change (optimal vs default): "
      f"${optimal_impact['total_cost'] - default_impact['total_cost']:,}")

# 4. Plot
plot_threshold_optimization(y_true, y_proba, metric='f1')

print("\nDone.")
