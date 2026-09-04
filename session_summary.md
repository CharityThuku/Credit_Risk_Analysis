# Credit Risk Analysis — Session Summary

## What this project is
A bootcamp-style credit risk classification assignment (LendingClub Q1 2019 loan data,
predicting default `high_risk` vs. `low_risk`). Started from `files.zip` (Downloads folder)
containing an AI-generated code-review/enhancement package — 6 files of template code and
docs that were never actually connected to the user's real notebooks. The user then supplied
the real notebooks, and we've spent this session finding and fixing real bugs, re-running
everything, and generating real charts.

All work lives in: `C:\Users\ckthu\Downloads\files_extracted\`

## Walkthrough progress (teaching plan, one step at a time)
User is new to ML metrics — walkthrough proceeds one concept/file at a time with check-ins.

- Done — Step 1: Big picture: imbalanced classification problem (loans, ~99.5% repay)
- Done — Step 2: Why accuracy misleads: confusion matrix, precision/recall/specificity/F1/balanced accuracy
- Done — Step 3: ROC-AUC: threshold-independent metric, how it's built, how to read it
- Done — Step 4: The 6 methods explained conceptually (resampling family vs. ensemble family), why ensemble beat resampling
- Done — Step 5: `credit_risk_metrics.py` code walkthrough (`evaluate_model()` line by line)
- Done — Step 6: `credit_risk_visualizations.py` / `COPY_PASTE_CODE.py` plotting functions — explained conceptually and generated real plots from corrected data
- Done — Step 7: `credit_risk_threshold_optimization.py`: threshold tuning + cost-benefit analysis on Easy Ensemble's real `y_proba` from `model_results.pkl` (no retraining needed). Found and fixed two more mislabeled-metric bugs in the template (see below) before trusting the numbers.
- Done — Step 8: `CODE_REVIEW_CHECKLIST.md` / `CREDIT_RISK_NEXT_STEPS.md` walkthrough, audited against the real notebooks. Found one more bug (hallucinated API param in the checklist itself, see below). See "Step 8" section below for the full audit and open questions for the user.
- User chose to pursue: cross-validation, hyperparameter tuning, SHAP explainability (skipped updating the stale docs for now).
- Done — Step 9: 5-fold cross-validation on all 6 methods. See results below -- a real, non-trivial finding here.
- Done — Step 10: hyperparameter tuning (Balanced RF, Easy Ensemble). See results below -- tuning barely moved the needle, which is itself a useful finding.
- In progress — Step 11: SHAP explainability
- Still pending: updating `EXECUTIVE_SUMMARY.md`/`README_IMPROVEMENTS.md`.
- Not yet done: updating `EXECUTIVE_SUMMARY.md` / `README_IMPROVEMENTS.md`, which still describe the old, buggy numbers — user explicitly deferred this to keep doing the technical walkthrough first

## Real bugs found in the original notebooks (all now fixed)
1. **Swapped `train_test_split` return values** in both notebooks — `X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=1)` had the tuple backwards, so models trained on 25% of data and were "tested" on 75%. Also missing `stratify=y`. Fixed to `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)`.
2. **SMOTEENN data leakage** — resampled on the *full* `X, y` instead of `X_train, y_train`.
3. **"Cluster Centroid Undersampling" section actually used `RandomUnderSampler`**, not `ClusterCentroids` — a mislabeled algorithm.
4. **"Balanced Random Forest" section actually used plain sklearn `RandomForestClassifier`**, not imblearn's `BalancedRandomForestClassifier` — invalidated the original narrative about that model's precision/recall behavior.
5. Minor: a garbled commented-out line reusing a stale `model` object in the SMOTEENN cell.
6. **`cost_benefit_analysis()` in `credit_risk_threshold_optimization.py` had two mislabeled business metrics** (found while running Step 7):
   - `accepted_rate` was computed as `(tp + fp) / total` — that's the rate of loans *flagged high-risk* (i.e. denied), the opposite of acceptance. Fixed to `(tn + fn) / total` (predicted-negative = approved population).
   - `default_rate_among_accepted` was computed as `fn / (tp + fn)` — that denominator is *all actual defaulters*, so the field was really just the false-negative rate (1 − recall), not a default rate among approved loans at all. At threshold 0.5 this printed **12.6%**, which would read to a business stakeholder as "1 in 8 approved loans defaults" — wildly misleading. Fixed to `fn / (tn + fn)`, the correct denominator (approved population); real value is **0.07%**.
7. **`CODE_REVIEW_CHECKLIST.md`'s Cluster Centroid Undersampling code sample calls a parameter that doesn't exist**: `ClusterCentroids(random_state=1, n_clusters=<minority_class_count>)`. Checked imblearn's real signature — `ClusterCentroids(*, sampling_strategy='auto', random_state=None, estimator=None, voting='auto')` — there is no `n_clusters` param; running the checklist's sample as-written raises `TypeError`. The actual notebook code (`ClusterCentroids(random_state=1)`) is fine and relies on the correct default (`sampling_strategy='auto'` balances to the minority count automatically) — this is a doc bug, not a notebook bug, but it's the same "hallucinated/wrong API" pattern as the earlier mislabeled-algorithm bugs. Not fixed in the checklist doc yet (low priority, reference-only file).

## Real corrected results (from `run_full_analysis.py`, the authoritative run)
| Method | Balanced Acc | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Naive Random Oversampling | 0.680 | 0.010 | 0.724 | 0.725 |
| SMOTE Oversampling | 0.662 | 0.010 | 0.690 | 0.713 |
| Cluster Centroid Undersampling (real) | 0.613 | 0.008 | 0.678 | 0.666 |
| SMOTEENN Sampling (leakage-free) | 0.655 | 0.009 | 0.724 | 0.720 |
| Balanced Random Forest (real) | 0.752 | 0.160 | 0.517 | 0.889 |
| Easy Ensemble | 0.906 | 0.067 | 0.874 | 0.964 |

Conclusion held and got *stronger* — Easy Ensemble is still the best model, now with real numbers
instead of numbers generated by a broken train/test split and two mislabeled algorithms.

**Noteworthy finding:** Easy Ensemble's top-5 feature importances include `issue_d_Mar-2019` /
`issue_d_Jan-2019` (loan issue month) — flagged as a possible temporal artifact (single-quarter
snapshot means earlier-issued loans had more time to default) rather than a genuine risk signal.
Worth a caveat in any writeup.

## Step 7 results: threshold optimization + cost-benefit (Easy Ensemble)
Run via `run_threshold_optimization.py` (new), using the corrected `credit_risk_threshold_optimization.py`.

Optimal decision thresholds by metric (default is 0.5):
| Metric | Optimal threshold | Score |
|---|---|---|
| F1 | 0.59 | 0.522 |
| Balanced accuracy | 0.49 | 0.910 |
| G-mean | 0.49 | 0.909 |

Cost-benefit comparison (cost per wrongly-denied good loan = $1,000; cost per approved bad loan = $5,000):
| Threshold | Precision | Recall | Acceptance rate | Default rate among accepted | Total cost |
|---|---|---|---|---|---|
| 0.3 | 0.005 | 1.000 | 0.0% | 0.00% | $17,118,000 |
| 0.4 | 0.005 | 1.000 | 6.8% | 0.00% | $15,940,000 |
| 0.5 (default) | 0.067 | 0.874 | 93.4% | 0.068% | $1,121,000 |
| 0.59 (F1-optimal) | 0.706 | 0.414 | 99.7% | 0.297% | $270,000 |
| 0.7 | 1.000 | 0.115 | 99.9% | 0.448% | $385,000 |

**Business takeaway:** with this cost ratio (denying a good customer costs 1/5 as much as approving a
defaulter), moving from the default 0.5 threshold to the F1-optimal 0.59 threshold *lowers total cost*
by $851,000, even though recall drops sharply (87% -> 41% of actual defaulters caught). This is because
at 0.5 the model over-flags: 1,066 good customers get wrongly denied per ~17k applicants, and that volume
of $1,000 false-positive costs outweighs the smaller number of $5,000 false-negative costs. The real
decision isn't "which threshold has the best F1" — it's "what are our actual denial and default costs,"
since the ranking of thresholds by total cost is sensitive to that ratio. Worth revisiting with real
lending economics if this were used for an actual business writeup, rather than the placeholder $1k/$5k
costs used here.

## Step 8: CODE_REVIEW_CHECKLIST.md / CREDIT_RISK_NEXT_STEPS.md audit
Went through both docs section by section against the real, fixed notebooks and scripts (not just taking the docs at face value, given the pattern of bugs found so far).

**Already satisfied by real work done:**
- Stratified train-test split with no leakage (Step 1-2 fixes)
- All 4 resampling methods correctly applied to `X_train`/`y_train` only, correct algorithms (Step 1-2 fixes)
- Balanced Random Forest + Easy Ensemble trained correctly, feature importance extracted (Step 4/6)
- Balanced accuracy + `classification_report_imbalanced` (precision, recall, specificity, geometric mean) computed **inside both notebooks** for every method
- ROC-AUC, F1, full 6-model comparison table, confusion matrix grid, feature importance chart — all done, but live in `run_full_analysis.py` / the 4 PNGs rather than inside the notebooks themselves
- Threshold optimization + cost-benefit analysis — done in Step 7

**Genuine gaps found (checked against actual code, not assumed):**
- `pd.get_dummies(X)` in both notebooks doesn't use `drop_first=True` as the checklist recommends. This only matters for the Logistic Regression models (tree-based models are unaffected by the resulting multicollinearity) — minor, not a correctness bug, just a style/statistics nicety. Not changed.
- ROC-AUC and F1 are computed in `run_full_analysis.py` but not inside the notebooks themselves — cosmetic gap only, the real numbers already exist and are correct.
- Genuinely not done anywhere: cross-validation (Step 7 in `CREDIT_RISK_NEXT_STEPS.md`), hyperparameter tuning (Step 8), learning curves, calibration analysis, isolation forest, SHAP/LIME. These are the "intermediate/advanced" tier — real additional analysis, not just documentation cleanup, and would take meaningful compute time on this 93MB dataset (Easy Ensemble alone re-trains 100 AdaBoost ensembles).

**Bug found in the checklist doc itself:** see bug #7 above (hallucinated `n_clusters` param on `ClusterCentroids`).

**Open question left for the user:** which (if any) of the intermediate/advanced enhancements in `CREDIT_RISK_NEXT_STEPS.md` (cross-validation, hyperparameter tuning, calibration, SHAP) to actually implement, since these are optional scope additions rather than corrections to existing work.

## Step 9 results: 5-fold cross-validation (all 6 methods)
Run via `run_cross_validation.py` (new), CV done strictly on `X_train` (imblearn `Pipeline` resamples inside each fold only, no leakage), `X_test` never touched. Purpose: check whether the single train/test split's numbers were representative.

| Method | Single-split Bal.Acc | CV Bal.Acc (mean +/- std) | Single-split ROC-AUC | CV ROC-AUC (mean +/- std) | Bal.Acc gap |
|---|---|---|---|---|---|
| Naive Random Oversampling | 0.679 | 0.677 +/- 0.045 | 0.725 | 0.731 +/- 0.036 | 0.1 sigma |
| SMOTE Oversampling | 0.662 | 0.654 +/- 0.052 | 0.713 | 0.717 +/- 0.043 | 0.1 sigma |
| Cluster Centroid Undersampling | 0.613 | 0.655 +/- 0.030 | 0.666 | 0.707 +/- 0.030 | -1.4 sigma |
| SMOTEENN Sampling | 0.655 | 0.660 +/- 0.040 | 0.720 | 0.714 +/- 0.043 | -0.1 sigma |
| Balanced Random Forest | 0.752 | 0.724 +/- 0.040 | 0.889 | 0.862 +/- 0.035 | 0.7 sigma |
| **Easy Ensemble** | **0.906** | **0.858 +/- 0.009** | 0.963 | 0.950 +/- 0.015 | **5.1 sigma** |

**Real finding, not just a validation formality:** every method's single-split balanced accuracy falls within ~1.4 standard deviations of its 5-fold CV mean -- *except* Easy Ensemble, which sits 5.1 standard deviations above its CV mean (0.906 vs. 0.858 +/- 0.009, a very tight CV distribution). That means the original 90.6% balanced-accuracy figure -- the one quoted as Easy Ensemble's headline number in `EXECUTIVE_SUMMARY.md` and the walkthrough so far -- looks like it came from a favorable test split for this specific metric, not a representative one. A more honest number is **~0.86**.

**The core conclusion still holds, though:** Easy Ensemble's ROC-AUC (the threshold-independent metric the recommendation is actually based on) is far more stable -- single-split 0.963 vs. CV mean 0.950 +/- 0.015, well within noise -- and still clearly the best of all 6 methods (next best, Balanced RF, CV means 0.862 +/- 0.035). So "Easy Ensemble is the best model" is robust; "Easy Ensemble achieves 90.6% balanced accuracy" is not -- that number should be corrected to ~86% (with the CV std noted) wherever it's quoted going forward, including the still-stale `EXECUTIVE_SUMMARY.md`.

**Why balanced accuracy is the noisy one here:** the test set only has 87 actual high-risk loans. Balanced accuracy's sensitivity term is estimated from that tiny sample, so it swings a lot between which 87 loans happen to land in the split. ROC-AUC integrates over all thresholds using all 17,205 predicted probabilities, so it's much less sensitive to exactly which few positive examples ended up in the held-out set.

## Step 10 results: hyperparameter tuning
Run via `run_hyperparameter_tuning.py` (new), `GridSearchCV` scored on ROC-AUC, 3-fold CV within `X_train`, evaluated once on the held-out `X_test`.

Note: `CREDIT_RISK_NEXT_STEPS.md`'s sample code (`GridSearchCV(EasyEnsembleClassifier(...), {'n_estimators': [...], 'max_depth': [...]}, ...)`) doesn't actually run -- `EasyEnsembleClassifier` has no top-level `max_depth` (it lives on the nested base estimator, reachable only via `estimator__estimator__max_depth`). Same "hallucinated/unreachable API" pattern as bug #7. Worked around it by tuning `n_estimators` only for Easy Ensemble.

| Model | Grid searched | Best params | Best CV ROC-AUC | Held-out test ROC-AUC | Untuned baseline test ROC-AUC |
|---|---|---|---|---|---|
| Balanced Random Forest | n_estimators in {100,200}, max_depth in {10, None} | n_estimators=200, max_depth=None | 0.862 | 0.883 | 0.889 (n_estimators=128) |
| Easy Ensemble | n_estimators in {50,100,150} | n_estimators=150 | 0.943 | 0.964 | 0.964 (n_estimators=100) |

**Finding:** tuning didn't meaningfully beat the original, essentially arbitrary settings (128 trees for Balanced RF, 100 estimators for Easy Ensemble) -- Easy Ensemble's tuned test ROC-AUC (0.964) is identical to baseline within rounding, and Balanced RF's tuned result (0.883) is actually *slightly below* its untuned baseline (0.889), well within noise. Read this as "the original settings were already in a flat part of the performance curve for this grid," not "the tuning failed" -- a real, if unglamorous, result worth reporting rather than skipping because it's not a dramatic improvement.

## Environment/tooling gotchas (if continuing on this machine)
- `imbalanced-learn` wasn't installed — now installed (`0.14.2`).
- `jupyter nbconvert` CLI doesn't resolve (PATH issue) — use `python3 -m nbconvert` instead.
- Notebooks' saved kernel (`mlenv`) doesn't exist locally — pass `--ExecutePreprocessor.kernel_name=python3`.
- `LoanStats_2019Q1.csv` was missing from the zip; downloaded from the public GitHub repo
  `npantfoerder/credit-risk-analysis` (redistributes the standard public LendingClub dataset used
  for this exact bootcamp assignment) into `files_extracted/`.
- `EasyEnsembleClassifier` has no top-level `feature_importances_` (unlike
  `BalancedRandomForestClassifier`) — had to average
  `eec.estimators_[i].named_steps['classifier'].feature_importances_` manually.
- Windows console is cp1252 — scripts printing "checkmark" characters crash with `UnicodeEncodeError`
  unless `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` is set first.
- Fixed an overlapping-label bug in `COPY_PASTE_CODE.py`'s `plot_precision_recall_tradeoff()`
  (points too close together) using union-find clustering + fanned-out stacked labels with leader lines.

## Current files in `files_extracted/`
- `credit_risk_resampling.ipynb`, `credit_risk_ensemble.ipynb` — fixed, re-executed, real outputs saved
- `LoanStats_2019Q1.csv` — the loan data (93MB)
- `run_full_analysis.py` — consolidated script: loads data, trains all 6 models with `predict_proba`
  extraction (which the original notebooks never had), builds comparison table, generates 4 plots,
  prints final recommendation
- `model_results.pkl` — saved y_true/y_pred/y_proba per model (reusable without retraining)
- `model_comparison.csv` — the table above
- `roc_comparison.png`, `confusion_matrices_all.png`, `feature_importance_comparison.png`,
  `precision_recall_tradeoff.png` — final, fixed visualizations
- `regenerate_pr_plot.py` — small helper to regenerate just the PR plot from the pickle
- `COPY_PASTE_CODE.py` — patched (label-overlap fix)
- `credit_risk_metrics.py` — `results_list` filled with placeholder variable names, superseded by `run_full_analysis.py`
- `credit_risk_threshold_optimization.py` — patched (`accepted_rate` / `default_rate_among_accepted` fix, see bug #6 above)
- `run_threshold_optimization.py` — new: runs Step 7 threshold tuning + cost-benefit analysis on Easy Ensemble's saved `y_proba`
- `threshold_comparison.csv` — new: business-impact table across thresholds 0.3-0.7
- `threshold_optimization_f1.png` — new: F1 vs. decision threshold plot
- `EXECUTIVE_SUMMARY.md`, `README_IMPROVEMENTS.md`, `CODE_REVIEW_CHECKLIST.md`, `CREDIT_RISK_NEXT_STEPS.md` — original docs, first two are stale (describe old buggy numbers)
