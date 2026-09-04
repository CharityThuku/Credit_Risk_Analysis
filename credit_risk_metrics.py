import pandas as pd
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc
)

# Template for collecting results from all 6 models
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Comprehensive evaluation of a single model
    
    Parameters:
    - y_true: actual labels
    - y_pred: predicted labels (from predict())
    - y_pred_proba: predicted probabilities (from predict_proba())[:, 1]
    - model_name: string identifier
    """
    
    # Binary metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate all metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # ROC-AUC (for high-risk class)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except:
        roc_auc = np.nan
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_acc,
        'Precision (High Risk)': precision,
        'Recall (High Risk)': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn
    }

# Usage example (replace with your actual predictions)
results_list = [
    evaluate_model(y_test, y_pred_naive, y_proba_naive, "Naive Random Oversampling"),
    evaluate_model(y_test, y_pred_smote, y_proba_smote, "SMOTE Oversampling"),
    evaluate_model(y_test, y_pred_cc, y_proba_cc, "Cluster Centroid Undersampling"),
    evaluate_model(y_test, y_pred_smoteenn, y_proba_smoteenn, "SMOTEENN Sampling"),
    evaluate_model(y_test, y_pred_brf, y_proba_brf, "Balanced Random Forest"),
    evaluate_model(y_test, y_pred_eec, y_proba_eec, "Easy Ensemble"),
]

# Create comparison DataFrame
results_df = pd.DataFrame(results_list)

# Sort by ROC-AUC (best overall metric for imbalanced data)
results_df_sorted = results_df.sort_values('ROC-AUC', ascending=False)

print("\n" + "="*100)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*100)
print(results_df_sorted.to_string(index=False))

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"Best ROC-AUC: {results_df_sorted.iloc[0]['Model']} ({results_df_sorted.iloc[0]['ROC-AUC']:.3f})")
print(f"Best Recall: {results_df_sorted.loc[results_df_sorted['Recall (High Risk)'].idxmax(), 'Model']}")
print(f"Best Precision: {results_df_sorted.loc[results_df_sorted['Precision (High Risk)'].idxmax(), 'Model']}")
print(f"Best F1 Score: {results_df_sorted.loc[results_df_sorted['F1 Score'].idxmax(), 'Model']}")

# Trade-off analysis
print("\n" + "="*80)
print("PRECISION-RECALL TRADE-OFF")
print("="*80)
for idx, row in results_df.iterrows():
    print(f"{row['Model']:35s} | Precision: {row['Precision (High Risk)']:.3f} | Recall: {row['Recall (High Risk)']:.3f} | F1: {row['F1 Score']:.3f}")
