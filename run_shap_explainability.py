"""
Step 11: SHAP explainability.

Balanced Random Forest gets exact, fast SHAP values via TreeExplainer (it's
a proper sklearn-style RandomForestClassifier subclass, fully supported).

Easy Ensemble is a Pipeline-of-100-AdaBoost-ensembles-of-decision-stumps --
not a structure TreeExplainer recognizes. Rather than force it, we use a
small-sample, model-agnostic Explainer (wraps predict_proba) with a
deliberately small background/explain set to keep runtime bounded, and
call out that these are approximate.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from data_loader import load_train_test

X_train, X_test, y_train, y_test, feature_names = load_train_test()

# ============================================================================
# 1. Balanced Random Forest -- exact TreeExplainer
# ============================================================================
print("Training Balanced Random Forest...")
brf = BalancedRandomForestClassifier(n_estimators=128, random_state=78)
brf.fit(X_train, y_train)

print("Computing SHAP values (TreeExplainer, exact) on a 1000-row test sample...")
t0 = time.time()
sample_idx = np.random.RandomState(1).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]

explainer = shap.TreeExplainer(brf)
shap_values = explainer.shap_values(X_sample)
# shap_values is [n_samples, n_features, n_classes] in recent shap for multiclass-capable trees
if isinstance(shap_values, list):
    sv_high_risk = shap_values[1]
elif shap_values.ndim == 3:
    sv_high_risk = shap_values[:, :, 1]
else:
    sv_high_risk = shap_values
print(f"  done in {time.time()-t0:.1f}s")

plt.figure()
shap.summary_plot(sv_high_risk, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.title("Balanced Random Forest -- SHAP feature impact on high_risk prediction")
plt.tight_layout()
plt.savefig('shap_balanced_rf_summary.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved shap_balanced_rf_summary.png")

mean_abs_shap = np.abs(sv_high_risk).mean(axis=0)
top10 = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])[:10]
print("\nTop 10 features by mean |SHAP value| (Balanced Random Forest):")
for name, val in top10:
    print(f"  {name}: {val:.4f}")

# ============================================================================
# 2. Easy Ensemble -- approximate, small-sample model-agnostic explainer
# ============================================================================
print("\nTraining Easy Ensemble...")
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)


def eec_high_risk_proba(X):
    idx = list(eec.classes_).index(1)
    return eec.predict_proba(X)[:, idx]


print("Computing approximate SHAP values (model-agnostic, small sample -- this is slow, kept deliberately tiny)...")
t0 = time.time()
rng = np.random.RandomState(1)
background_idx = rng.choice(len(X_train), size=50, replace=False)
explain_idx = rng.choice(len(X_test), size=30, replace=False)
background = shap.sample(X_train.iloc[background_idx], 50)

eec_explainer = shap.Explainer(eec_high_risk_proba, background, feature_names=feature_names)
eec_shap_values = eec_explainer(X_test.iloc[explain_idx], max_evals=500)
print(f"  done in {time.time()-t0:.1f}s (n=30 explained rows -- approximate, not for production use)")

plt.figure()
shap.summary_plot(eec_shap_values.values, X_test.iloc[explain_idx], feature_names=feature_names,
                   show=False, max_display=15)
plt.title("Easy Ensemble -- approximate SHAP (n=30 sample)")
plt.tight_layout()
plt.savefig('shap_easy_ensemble_summary_APPROX.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved shap_easy_ensemble_summary_APPROX.png (approximate -- small sample)")

mean_abs_shap_eec = np.abs(eec_shap_values.values).mean(axis=0)
top10_eec = sorted(zip(feature_names, mean_abs_shap_eec), key=lambda x: -x[1])[:10]
print("\nTop 10 features by mean |SHAP value| (Easy Ensemble, approximate, n=30):")
for name, val in top10_eec:
    print(f"  {name}: {val:.4f}")

print("\nDone.")
