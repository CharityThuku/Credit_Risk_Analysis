"""
Step 9: 5-fold stratified cross-validation on the training set for all 6
methods, to check whether the single train/test split's numbers in
model_comparison.csv were representative or just a lucky/unlucky split.

CV is run strictly within X_train (never touches X_test). For the 4
resampling methods, an imblearn Pipeline resamples inside each training
fold only -- no leakage into the held-out fold.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from data_loader import load_train_test

X_train, X_test, y_train, y_test, feature_names = load_train_test()
print(f"CV runs on X_train only: {X_train.shape}")

methods = {
    'Naive Random Oversampling': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=1)),
        ('clf', LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)),
    ]),
    'SMOTE Oversampling': ImbPipeline([
        ('sampler', SMOTE(random_state=1)),
        ('clf', LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)),
    ]),
    'Cluster Centroid Undersampling': ImbPipeline([
        ('sampler', ClusterCentroids(random_state=1)),
        ('clf', LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)),
    ]),
    'SMOTEENN Sampling': ImbPipeline([
        ('sampler', SMOTEENN(random_state=1)),
        ('clf', LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)),
    ]),
    'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=128, random_state=78),
    'Easy Ensemble': EasyEnsembleClassifier(n_estimators=100, random_state=1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scoring = ['balanced_accuracy', 'roc_auc', 'f1']

rows = []
for name, estimator in methods.items():
    t0 = time.time()
    print(f"\nRunning 5-fold CV: {name} ...")
    scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - t0
    row = {'method': name}
    for metric in scoring:
        vals = scores[f'test_{metric}']
        row[f'{metric}_mean'] = vals.mean()
        row[f'{metric}_std'] = vals.std()
    row['seconds'] = round(elapsed, 1)
    rows.append(row)
    print(f"  balanced_accuracy: {row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_std']:.3f}")
    print(f"  roc_auc:           {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}")
    print(f"  f1:                {row['f1_mean']:.3f} +/- {row['f1_std']:.3f}")
    print(f"  ({elapsed:.1f}s)")

cv_df = pd.DataFrame(rows)
cv_df.to_csv('cross_validation_results.csv', index=False)
print("\nSaved cross_validation_results.csv")
print(cv_df.to_string(index=False))
