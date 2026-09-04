import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# ============================================================================
# 1. ROC-AUC CURVE COMPARISON (Best single metric for imbalanced data)
# ============================================================================

def plot_roc_curves(models_dict):
    """
    Plot ROC curves for all models on one chart
    
    Parameters:
    models_dict: {
        'model_name': {'y_true': array, 'y_proba': array},
        ...
    }
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, data in models_dict.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC-AUC Comparison: All 6 Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ ROC curves saved to 'roc_comparison.png'")

# ============================================================================
# 2. CONFUSION MATRIX HEATMAPS (Side-by-side comparison)
# ============================================================================

def plot_confusion_matrices(predictions_dict, y_test):
    """
    Create a 2x3 grid of confusion matrices for all 6 models
    
    Parameters:
    predictions_dict: {
        'model_name': y_predictions_array,
        ...
    }
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    model_names = list(predictions_dict.keys())
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                    ax=axes[idx], cbar=False,
                    xticklabels=['Low Risk', 'High Risk'],
                    yticklabels=['Low Risk', 'High Risk'])
        
        axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xlabel('Predicted', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Confusion matrices saved to 'confusion_matrices.png'")

# ============================================================================
# 3. FEATURE IMPORTANCE (Tree-based models only: Balanced RF & Easy Ensemble)
# ============================================================================

def plot_feature_importance(brf_model, eec_model, feature_names, top_n=15):
    """
    Compare feature importance: Balanced Random Forest vs Easy Ensemble
    
    Parameters:
    - brf_model: fitted BalancedRandomForestClassifier
    - eec_model: fitted EasyEnsembleClassifier
    - feature_names: list of feature column names
    - top_n: how many top features to display
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Balanced Random Forest Feature Importance
    brf_importance = pd.Series(brf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    brf_importance.head(top_n).plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Balanced Random Forest - Top 15 Features', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature Importance', fontsize=11)
    axes[0].invert_yaxis()
    
    # Easy Ensemble Feature Importance
    eec_importance = pd.Series(eec_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    eec_importance.head(top_n).plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Easy Ensemble - Top 15 Features', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Feature Importance', fontsize=11)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Feature importance saved to 'feature_importance.png'")
    
    # Print top 5 important features
    print("\n" + "="*80)
    print("TOP 5 IMPORTANT FEATURES FOR DETECTING HIGH-RISK LOANS")
    print("="*80)
    print("\nBalanced Random Forest:")
    print(brf_importance.head(5).to_string())
    print("\nEasy Ensemble:")
    print(eec_importance.head(5).to_string())

# ============================================================================
# 4. PRECISION-RECALL TRADE-OFF VISUALIZATION
# ============================================================================

def plot_precision_recall_comparison(results_df):
    """
    Scatter plot showing precision vs recall for all models
    Helps visualize the trade-off
    
    Parameters:
    results_df: DataFrame with columns ['Model', 'Precision (High Risk)', 'Recall (High Risk)']
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color code: ensemble methods in one color, resampling in another
    colors = {
        'Naive Random Oversampling': 'red',
        'SMOTE Oversampling': 'red',
        'Cluster Centroid Undersampling': 'red',
        'SMOTEENN Sampling': 'orange',
        'Balanced Random Forest': 'green',
        'Easy Ensemble': 'blue'
    }
    
    for idx, row in results_df.iterrows():
        color = colors.get(row['Model'], 'gray')
        ax.scatter(row['Precision (High Risk)'], row['Recall (High Risk)'], 
                  s=300, alpha=0.6, color=color, edgecolors='black', linewidth=2)
        
        # Add label
        ax.annotate(row['Model'], 
                   xy=(row['Precision (High Risk)'], row['Recall (High Risk)']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Ideal point (Precision=1, Recall=1)
    ax.scatter(1, 1, marker='*', s=1000, color='gold', edgecolors='black', linewidth=2, label='Ideal Model')
    
    ax.set_xlabel('Precision (High Risk)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (High Risk)', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-Off: All 6 Models', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add quadrant labels
    ax.text(0.5, 0.95, 'High Recall\nLow Precision', ha='center', va='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.95, 0.95, 'High Recall\nHigh Precision', ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Precision-recall trade-off saved to 'precision_recall_tradeoff.png'")

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

print("""
To use these functions in your notebook:

1. For ROC curves:
   plot_roc_curves({
       'Naive Random Oversampling': {'y_true': y_test, 'y_proba': y_proba_naive},
       'SMOTE Oversampling': {'y_true': y_test, 'y_proba': y_proba_smote},
       # ... add all 6 models
   })

2. For confusion matrices:
   plot_confusion_matrices({
       'Naive Random Oversampling': y_pred_naive,
       'SMOTE Oversampling': y_pred_smote,
       # ... add all 6 models
   }, y_test)

3. For feature importance:
   plot_feature_importance(brf_model, eec_model, X_train.columns, top_n=15)

4. For precision-recall trade-off:
   plot_precision_recall_comparison(results_df)  # Your results DataFrame
""")
