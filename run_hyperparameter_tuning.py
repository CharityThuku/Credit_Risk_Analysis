"""
Step 10: Hyperparameter tuning for the two ensemble models (Balanced Random
Forest, Easy Ensemble) via GridSearchCV, scored on ROC-AUC (the metric the
project's final recommendation is based on).

Note: CREDIT_RISK_NEXT_STEPS.md's sample code
    GridSearchCV(EasyEnsembleClassifier(random_state=1),
                 {'n_estimators': [...], 'max_depth': [...]}, ...)
doesn't actually work -- EasyEnsembleClassifier has no top-level max_depth
(its base estimator does, but reaching it needs the nested key
'estimator__estimator__max_depth'). Kept the EasyEnsemble grid to
n_estimators only rather than fight that nesting for a teaching exercise.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from data_loader import load_train_test

X_train, X_test, y_train, y_test, feature_names = load_train_test()
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

results = {}

print("Tuning Balanced Random Forest (n_estimators x max_depth)...")
t0 = time.time()
brf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
}
brf_search = GridSearchCV(
    BalancedRandomForestClassifier(random_state=78),
    brf_grid, scoring='roc_auc', cv=cv, n_jobs=-1,
)
brf_search.fit(X_train, y_train)
print(f"  Best params: {brf_search.best_params_}")
print(f"  Best CV ROC-AUC: {brf_search.best_score_:.4f}  ({time.time()-t0:.1f}s)")
brf_test_auc = brf_search.score(X_test, y_test)
print(f"  Held-out test ROC-AUC with best params: {brf_test_auc:.4f}")
results['Balanced Random Forest'] = {
    'best_params': brf_search.best_params_,
    'best_cv_roc_auc': brf_search.best_score_,
    'test_roc_auc': brf_test_auc,
}

print("\nTuning Easy Ensemble (n_estimators only, see note above)...")
t0 = time.time()
eec_grid = {
    'n_estimators': [50, 100, 150],
}
eec_search = GridSearchCV(
    EasyEnsembleClassifier(random_state=1),
    eec_grid, scoring='roc_auc', cv=cv, n_jobs=-1,
)
eec_search.fit(X_train, y_train)
print(f"  Best params: {eec_search.best_params_}")
print(f"  Best CV ROC-AUC: {eec_search.best_score_:.4f}  ({time.time()-t0:.1f}s)")
eec_test_auc = eec_search.score(X_test, y_test)
print(f"  Held-out test ROC-AUC with best params: {eec_test_auc:.4f}")
results['Easy Ensemble'] = {
    'best_params': eec_search.best_params_,
    'best_cv_roc_auc': eec_search.best_score_,
    'test_roc_auc': eec_test_auc,
}

rows = []
for name, r in results.items():
    row = {'method': name, **r['best_params'], 'best_cv_roc_auc': r['best_cv_roc_auc'], 'test_roc_auc': r['test_roc_auc']}
    rows.append(row)
pd.DataFrame(rows).to_csv('hyperparameter_tuning_results.csv', index=False)
print("\nSaved hyperparameter_tuning_results.csv")

print("\nComparison to un-tuned baseline (from model_comparison.csv):")
baseline = pd.read_csv('model_comparison.csv', index_col=0)
print(baseline.loc[['Balanced Random Forest', 'Easy Ensemble'], ['ROC-AUC']])
