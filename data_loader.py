"""
Shared data loading/preprocessing, factored out of run_full_analysis.py so
Steps 9-11 (cross-validation, hyperparameter tuning, SHAP) use the exact
same train/test split and encoding instead of re-deriving it and risking drift.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
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


def load_train_test(binary_target=True):
    """
    Returns X_train, X_test, y_train, y_test, feature_names.
    If binary_target, y is 0/1 (1 = high_risk); otherwise the original strings.
    Identical preprocessing and split (random_state=1, stratify=y) as
    run_full_analysis.py / the fixed notebooks.
    """
    df = pd.read_csv(Path('LoanStats_2019Q1.csv'), skiprows=1)[:-2]
    df = df.loc[:, COLUMNS].copy()
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
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

    if binary_target:
        y_train = (y_train == 'high_risk').astype(int)
        y_test = (y_test == 'high_risk').astype(int)

    return X_train, X_test, y_train, y_test, feature_names
