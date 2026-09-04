import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # headless: save PNGs, don't try to open a GUI window

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

import matplotlib.pyplot as plt

from COPY_PASTE_CODE import (
    create_comparison_table,
    plot_all_confusion_matrices,
    plot_roc_comparison,
    plot_precision_recall_tradeoff,
    print_final_recommendation,
)

# ============================================================================
# 1. LOAD & CLEAN DATA (identical to the fixed notebooks)
# ============================================================================

columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

print("Loading data...")
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()
df = df.dropna(axis='columns', how='all')
df = df.dropna()
df = df.loc[df['loan_status'] != 'Issued']

df['int_rate'] = df['int_rate'].str.replace('%', '').astype('float') / 100

df = df.replace({'Current': 'low_risk'})
df = df.replace(dict.fromkeys(
    ['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk'
))
df.reset_index(inplace=True, drop=True)

X = pd.get_dummies(df.drop('loan_status', axis=1))
y = df['loan_status']
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
y_test_bin = (y_test == 'high_risk').astype(int).values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Test class balance:\n{y_test.value_counts()}")


def get_proba(model, X):
    idx = list(model.classes_).index('high_risk')
    return model.predict_proba(X)[:, idx]


results = {}

# ============================================================================
# 2. TRAIN ALL 6 MODELS
# ============================================================================

print("\n[1/6] Naive Random Oversampling...")
ros = RandomOverSampler(random_state=1)
X_res, y_res = ros.fit_resample(X_train, y_train)
lr = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)
results['Naive Random Oversampling'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(lr, X_test),
}

print("[2/6] SMOTE Oversampling...")
X_res, y_res = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)
lr = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)
results['SMOTE Oversampling'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(lr, X_test),
}

print("[3/6] Cluster Centroid Undersampling (slow)...")
cc = ClusterCentroids(random_state=1)
X_res, y_res = cc.fit_resample(X_train, y_train)
lr = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)
results['Cluster Centroid Undersampling'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(lr, X_test),
}

print("[4/6] SMOTEENN Sampling (slow)...")
smote_enn = SMOTEENN(random_state=1)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)
lr = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)
results['SMOTEENN Sampling'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(lr, X_test),
}

print("[5/6] Balanced Random Forest...")
brf = BalancedRandomForestClassifier(n_estimators=128, random_state=78)
brf.fit(X_train, y_train)
y_pred = brf.predict(X_test)
results['Balanced Random Forest'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(brf, X_test),
}

print("[6/6] Easy Ensemble (slow)...")
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)
y_pred = eec.predict(X_test)
results['Easy Ensemble'] = {
    'y_true': y_test_bin,
    'y_pred': (y_pred == 'high_risk').astype(int),
    'y_proba': get_proba(eec, X_test),
}

# Save raw results so later steps (threshold optimization) don't need to retrain
with open('model_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'feature_names': list(feature_names)}, f)
print("\nSaved predictions to model_results.pkl")

# ============================================================================
# 3. COMPARISON TABLE
# ============================================================================

comparison_df = create_comparison_table(results)
comparison_df.to_csv('model_comparison.csv')
print("Saved comparison table to model_comparison.csv")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

print("\nGenerating confusion matrix grid...")
plot_all_confusion_matrices({k: v['y_pred'] for k, v in results.items()}, y_test_bin)

print("Generating ROC curve comparison...")
plot_roc_comparison({k: {'y_true': v['y_true'], 'y_proba': v['y_proba']} for k, v in results.items()})

print("Generating feature importance comparison...")
# NOTE: EasyEnsembleClassifier has no top-level feature_importances_ (unlike
# BalancedRandomForestClassifier) -- COPY_PASTE_CODE.py's plot_feature_importance()
# would crash on it. Each of its 100 estimators is a Pipeline(sampler, AdaBoostClassifier);
# we average the AdaBoostClassifier's feature_importances_ across all of them instead.
eec_importance_values = np.mean(
    [pipe.named_steps['classifier'].feature_importances_ for pipe in eec.estimators_],
    axis=0,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

brf_importance = pd.Series(brf.feature_importances_, index=feature_names).sort_values(ascending=False)
brf_importance.head(15).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Balanced Random Forest - Top Features', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].invert_yaxis()

eec_importance = pd.Series(eec_importance_values, index=feature_names).sort_values(ascending=False)
eec_importance.head(15).plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Easy Ensemble - Top Features (avg. over 100 base learners)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=11)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Top 5 features (Balanced Random Forest):")
print(brf_importance.head(5).to_string())
print("Top 5 features (Easy Ensemble):")
print(eec_importance.head(5).to_string())

print("Generating precision-recall trade-off plot...")
plot_precision_recall_tradeoff(comparison_df.reset_index())

# ============================================================================
# 5. FINAL RECOMMENDATION
# ============================================================================

print_final_recommendation(comparison_df)
