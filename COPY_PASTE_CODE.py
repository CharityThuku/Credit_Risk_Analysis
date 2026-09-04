# ============================================================================
# CREDIT RISK ANALYSIS — COPY-PASTE CODE SNIPPETS
# ============================================================================
# Add these to your notebooks to enhance analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc
)

# ============================================================================
# SNIPPET 1: Comprehensive Metrics for ANY Model
# ============================================================================
# Run this after each model's predictions
# Usage: print_model_metrics("Model Name", y_test, y_pred, y_proba)

def print_model_metrics(model_name, y_true, y_pred, y_proba=None):
    """
    Print all important metrics for a classification model
    """
    print("\n" + "="*70)
    print(f"MODEL: {model_name}")
    print("="*70)
    
    # Extract confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
    
    # Print results
    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC:           {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp:5d} | FP={fp:5d}")
    print(f"  FN={fn:5d} | TN={tn:5d}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Low Risk', 'High Risk']))

# Example usage:
# print_model_metrics("Naive Random Oversampling", y_test, y_pred_naive, y_proba_naive)


# ============================================================================
# SNIPPET 2: Create Results Summary Table (Combine all 6 models)
# ============================================================================

def create_comparison_table(results_dict):
    """
    Create a comparison table from all models
    
    results_dict format:
    {
        'Model Name': {'y_true': array, 'y_pred': array, 'y_proba': array},
        ...
    }
    """
    
    results_list = []
    
    for model_name, data in results_dict.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_proba = data.get('y_proba')
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
        
        results_list.append({
            'Model': model_name,
            'Accuracy': balanced_acc,
            'Precision': precision,
            'Recall': recall,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        })
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index('Model')
    
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*100)
    print(results_df.round(4).to_string())
    
    return results_df

# Example usage:
# results_dict = {
#     'Naive Random Oversampling': {'y_true': y_test, 'y_pred': y_pred_naive, 'y_proba': y_proba_naive},
#     'SMOTE Oversampling': {'y_true': y_test, 'y_pred': y_pred_smote, 'y_proba': y_proba_smote},
#     # ... etc for all 6 models
# }
# comparison_df = create_comparison_table(results_dict)


# ============================================================================
# SNIPPET 3: Plot Confusion Matrices (All 6 in one grid)
# ============================================================================

def plot_all_confusion_matrices(predictions_dict, y_test, figsize=(15, 10)):
    """
    Plot 2x3 grid of confusion matrices
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                    ax=axes[idx], cbar=False,
                    xticklabels=['Low Risk', 'High Risk'],
                    yticklabels=['Low Risk', 'High Risk'])
        
        axes[idx].set_title(model_name, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xlabel('Predicted', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Confusion matrices saved")

# Example usage:
# predictions = {
#     'Naive Random': y_pred_naive,
#     'SMOTE': y_pred_smote,
#     # ... etc
# }
# plot_all_confusion_matrices(predictions, y_test)


# ============================================================================
# SNIPPET 4: Plot ROC-AUC Curves (All 6 models)
# ============================================================================

def plot_roc_comparison(models_dict, figsize=(10, 8)):
    """
    Plot ROC curves for all models
    
    models_dict: {'model_name': {'y_true': array, 'y_proba': array}}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, data in models_dict.items():
        y_true = data['y_true']
        y_proba = data['y_proba']
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2.5)
        except:
            print(f"⚠️ Could not plot ROC for {model_name}")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC-AUC Comparison: All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ ROC curves saved")

# Example usage:
# models_dict = {
#     'Naive Random': {'y_true': y_test, 'y_proba': y_proba_naive},
#     'Easy Ensemble': {'y_true': y_test, 'y_proba': y_proba_eec},
#     # ... etc
# }
# plot_roc_comparison(models_dict)


# ============================================================================
# SNIPPET 5: Feature Importance (For tree-based models)
# ============================================================================

def plot_feature_importance(model1, model2, feature_names, 
                            model1_name, model2_name, top_n=15):
    """
    Compare feature importance from two tree-based models
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model 1
    imp1 = pd.Series(model1.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp1.head(top_n).plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title(f'{model1_name} - Top Features', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].invert_yaxis()
    
    # Model 2
    imp2 = pd.Series(model2.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp2.head(top_n).plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title(f'{model2_name} - Top Features', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Feature importance comparison saved")
    
    print(f"\nTop 5 features ({model1_name}):")
    print(imp1.head(5).to_string())
    print(f"\nTop 5 features ({model2_name}):")
    print(imp2.head(5).to_string())

# Example usage:
# plot_feature_importance(brf_model, eec_model, X_train.columns,
#                        'Balanced Random Forest', 'Easy Ensemble', top_n=15)


# ============================================================================
# SNIPPET 6: Precision-Recall Trade-off Visualization
# ============================================================================

def plot_precision_recall_tradeoff(results_df):
    """
    Scatter plot: Precision vs Recall for all models.
    Imbalanced-data models often land almost on top of each other (e.g. several
    resampling methods all near precision~0.01), so direct labels would overlap
    into unreadable text. Points closer than CLUSTER_DIST are grouped and their
    labels fanned into a stacked list with thin leader lines back to the dot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        'Naive Random Oversampling': '#e74c3c',
        'SMOTE Oversampling': '#e74c3c',
        'Cluster Centroid Undersampling': '#e74c3c',
        'SMOTEENN Sampling': '#f39c12',
        'Balanced Random Forest': '#2ecc71',
        'Easy Ensemble': '#3498db'
    }

    rows = [(row['Model'], row['Precision'], row['Recall']) for _, row in results_df.iterrows()]

    for model, x, y in rows:
        color = colors.get(model, 'gray')
        ax.scatter(x, y, s=400, alpha=0.75, color=color, edgecolors='black', linewidth=2, zorder=3)

    # --- Union-find clustering of points that are too close for direct labels ---
    CLUSTER_DIST = 0.08  # data units; Precision/Recall both span 0-1
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dx = rows[i][1] - rows[j][1]
            dy = rows[i][2] - rows[j][2]
            if (dx ** 2 + dy ** 2) ** 0.5 < CLUSTER_DIST:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    for members in clusters.values():
        if len(members) == 1:
            model, x, y = rows[members[0]]
            ax.annotate(model, xy=(x, y), xytext=(8, 8), textcoords='offset points', fontsize=9)
            continue

        # Fan overlapping labels into a vertical stack to the right of the cluster,
        # highest recall on top, each with a thin leader line back to its real point.
        members_sorted = sorted(members, key=lambda i: -rows[i][2])
        label_x = max(rows[i][1] for i in members) + 0.10
        label_y_top = max(rows[i][2] for i in members) + 0.03
        step = 0.045
        for k, i in enumerate(members_sorted):
            model, x, y = rows[i]
            label_y = label_y_top - k * step
            ax.annotate(
                model, xy=(x, y), xytext=(label_x, label_y),
                textcoords='data', fontsize=9, va='center',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, shrinkA=5, shrinkB=5),
            )

    # Ideal point
    ax.scatter(1, 1, marker='*', s=1500, color='gold', edgecolors='black',
              linewidth=2, label='Ideal Model')

    ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Precision-Recall trade-off plot saved")

# Example usage:
# plot_precision_recall_tradeoff(comparison_df)


# ============================================================================
# SNIPPET 7: Quick Summary (Run this at the END of your notebook)
# ============================================================================

def print_final_recommendation(comparison_df):
    """
    Print executive summary with recommendation
    """
    best_roc_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
    best_recall = comparison_df.loc[comparison_df['Recall'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    print(f"\n🏆 Best Overall Model (ROC-AUC): {best_roc_auc.name}")
    print(f"   Score: {best_roc_auc['ROC-AUC']:.4f}")
    
    print(f"\n🎯 Best at Catching Defaults (Recall): {best_recall.name}")
    print(f"   Recall: {best_recall['Recall']:.4f}")
    
    print(f"\n⚡ Best Balance (F1 Score): {best_f1.name}")
    print(f"   F1: {best_f1['F1 Score']:.4f}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Use '{best_roc_auc.name}' for credit risk prediction")
    print(f"   ")
    print(f"   Rationale:")
    print(f"   - Highest ROC-AUC ({best_roc_auc['ROC-AUC']:.4f}): Best overall discrimination")
    print(f"   - High Recall ({best_roc_auc['Recall']:.4f}): Catches most defaults")
    print(f"   - Accuracy ({best_roc_auc['Accuracy']:.4f}): Reliable predictions")
    print(f"\n   Next steps: Optimize decision threshold to maximize F1 or minimize cost")
    print("="*80)

# Example usage:
# print_final_recommendation(comparison_df)


# ============================================================================
# FULL INTEGRATION EXAMPLE
# ============================================================================

"""
Here's how to integrate all these snippets into your notebook:

# After training all 6 models, collect predictions:
all_models = {
    'Naive Random Oversampling': {
        'y_true': y_test,
        'y_pred': y_pred_naive,
        'y_proba': y_proba_naive
    },
    'SMOTE Oversampling': {
        'y_true': y_test,
        'y_pred': y_pred_smote,
        'y_proba': y_proba_smote
    },
    # ... add all 6 models
}

# Create comparison table
comparison_df = create_comparison_table(all_models)

# Plot confusion matrices
plot_all_confusion_matrices({k: v['y_pred'] for k, v in all_models.items()}, y_test)

# Plot ROC curves
plot_roc_comparison({k: {'y_true': v['y_true'], 'y_proba': v['y_proba']} 
                     for k, v in all_models.items()})

# Feature importance (if using Balanced RF and Easy Ensemble)
plot_feature_importance(brf_model, eec_model, X_train.columns,
                       'Balanced Random Forest', 'Easy Ensemble')

# Precision-Recall trade-off
plot_precision_recall_tradeoff(comparison_df)

# Print recommendation
print_final_recommendation(comparison_df)
"""
