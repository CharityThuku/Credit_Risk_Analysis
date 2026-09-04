import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# THRESHOLD TUNING
# ============================================================================
# Default threshold is 0.5, but for imbalanced data this is often suboptimal
# Tuning threshold shifts the precision-recall trade-off

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal decision threshold for a classifier
    
    Parameters:
    - y_true: actual labels (0/1)
    - y_proba: predicted probabilities from predict_proba()[:, 1]
    - metric: 'f1' (default), 'balanced_accuracy', or 'g_mean'
    
    Returns:
    - optimal_threshold: threshold that maximizes chosen metric
    - scores: metric value at each threshold
    - thresholds: tested thresholds (0.01 to 0.99)
    """
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = (tp / (tp + fn) + tn / (tn + fp)) / 2  # (sensitivity + specificity) / 2
        elif metric == 'g_mean':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            score = np.sqrt(sensitivity * specificity)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    best_score = scores[optimal_idx]
    
    return optimal_threshold, np.array(scores), thresholds, best_score

# ============================================================================
# COST-BENEFIT ANALYSIS
# ============================================================================
# In credit risk, FP (false positive/wrongly denied) costs differ from FN (false negative/bad loan)

def cost_benefit_analysis(y_true, y_proba, 
                         cost_fp=1000,  # Cost of denying a good customer
                         cost_fn=5000,  # Cost of accepting a defaulter
                         threshold=0.5):
    """
    Calculate business impact of model predictions
    
    Parameters:
    - y_true: actual labels
    - y_proba: predicted probabilities
    - cost_fp: $ cost per false positive (wrongly denied good loan)
    - cost_fn: $ cost per false negative (bad loan approved)
    - threshold: decision threshold
    
    Returns:
    Dictionary with business metrics
    """
    
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Financial impact
    cost_false_positives = fp * cost_fp
    cost_false_negatives = fn * cost_fn
    total_cost = cost_false_positives + cost_false_negatives
    
    # Operational metrics
    # y_pred == 1 means "flagged high-risk" -> denied. Accepted loans are the
    # predicted-negative population (tn + fn), not (tp + fp) as originally
    # written here (that was the denial rate, mislabeled as acceptance).
    total_applicants = len(y_true)
    accepted_rate = (tn + fn) / total_applicants
    default_rate_among_accepted = fn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'threshold': threshold,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'cost_false_positives': cost_false_positives,
        'cost_false_negatives': cost_false_negatives,
        'total_cost': total_cost,
        'acceptance_rate': accepted_rate,
        'default_rate_among_accepted': default_rate_among_accepted,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }

# ============================================================================
# COMPARE THRESHOLDS
# ============================================================================

def compare_thresholds(y_true, y_proba, thresholds_to_test=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Compare business impact across different thresholds
    """
    
    results = []
    for threshold in thresholds_to_test:
        result = cost_benefit_analysis(y_true, y_proba, threshold=threshold)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*120)
    print("THRESHOLD COMPARISON: Business Impact Analysis")
    print("="*120)
    print(results_df[['threshold', 'precision', 'recall', 'acceptance_rate', 
                       'default_rate_among_accepted', 'total_cost']].to_string(index=False))
    
    return results_df

# ============================================================================
# VISUALIZATION: Threshold vs Metrics
# ============================================================================

def plot_threshold_optimization(y_true, y_proba, metric='f1'):
    """
    Plot how performance changes with decision threshold
    """
    
    optimal_threshold, scores, thresholds, best_score = find_optimal_threshold(
        y_true, y_proba, metric=metric
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, scores, linewidth=2.5, color='steelblue')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Optimal: {optimal_threshold:.2f} ({metric.upper()} = {best_score:.3f})')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='Default threshold (0.5)')
    
    ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Finding Optimal Threshold ({metric.upper()})', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'threshold_optimization_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Threshold optimization plot saved to 'threshold_optimization_{metric}.png'")
    print(f"\nOptimal threshold for {metric}: {optimal_threshold:.2f}")
    print(f"Score at optimal threshold: {best_score:.3f}")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

example_usage = """
# In your notebook, after training Easy Ensemble:

# 1. Find optimal threshold
optimal_threshold, scores, thresholds, best_f1 = find_optimal_threshold(
    y_test, y_proba_eec, metric='f1'
)
print(f"Optimal threshold: {optimal_threshold:.2f} (F1={best_f1:.3f})")

# 2. Compare thresholds with cost-benefit analysis
results_df = compare_thresholds(y_test, y_proba_eec, 
                                 thresholds_to_test=[0.3, 0.4, 0.5, 0.6, 0.7])

# 3. Check business impact at optimal threshold
impact = cost_benefit_analysis(y_test, y_proba_eec, 
                               cost_fp=1000,  # $1k per wrongly denied customer
                               cost_fn=5000,  # $5k per bad loan accepted
                               threshold=optimal_threshold)
print(f"\\nBusiness Impact at threshold {optimal_threshold}:")
print(f"  Total Cost: ${impact['total_cost']:,}")
print(f"  Acceptance Rate: {impact['acceptance_rate']:.1%}")
print(f"  Default Rate (among accepted): {impact['default_rate_among_accepted']:.1%}")

# 4. Visualize threshold optimization
plot_threshold_optimization(y_test, y_proba_eec, metric='f1')
"""

print(example_usage)
