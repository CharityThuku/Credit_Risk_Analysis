# Credit Risk Analysis — Next Steps & Advanced Improvements

## 📊 QUICK WINS (Add to notebooks this week)

### 1. **Add ROC-AUC Metrics**
- [ ] Calculate ROC-AUC for all 6 models
- [ ] Plot ROC curves on single chart
- [ ] **Why:** ROC-AUC is THE metric for imbalanced classification (better than accuracy)
- **Code template:** See `credit_risk_visualizations.py`

### 2. **Fix Missing Metrics**
- [ ] Add F1 scores alongside precision/recall
- [ ] Include specificity (true negative rate)
- [ ] Include balanced accuracy (not just accuracy)
- **Why:** Gives complete picture of trade-offs

```python
from sklearn.metrics import balanced_accuracy_score, f1_score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

### 3. **Create Confusion Matrix Grid**
- [ ] Make 2x3 subplot showing all 6 confusion matrices
- [ ] Normalize by row for visual comparison
- **Why:** Easy side-by-side comparison of TP/TN/FP/FN

### 4. **Feature Importance Analysis**
- [ ] Extract top 15 features from Balanced RF + Easy Ensemble
- [ ] Create bar charts comparing them
- **Why:** Answers "which factors actually matter for defaults?"
- **Tools:** `feature_importances_` attribute on tree models
- **Expected insight:** Likely: interest_rate, term, loan_amount, income

---

## 🎯 INTERMEDIATE ENHANCEMENTS (Next 2 weeks)

### 5. **Threshold Optimization**
**Problem:** You're using 0.5 threshold for all models, but that's arbitrary

**Solution:**
```python
from sklearn.metrics import precision_recall_curve

# Find threshold that maximizes F1 score
optimal_threshold, _, _, best_f1 = find_optimal_threshold(
    y_test, y_proba_eec, metric='f1'
)
# Likely result: 0.25-0.35 (lower threshold → catch more defaults)
```

**Impact:** Easy Ensemble might go from 76% recall to 85%+ recall at optimal threshold

### 6. **Cost-Benefit Analysis**
**Real-world question:** What's the business impact?

```python
# Scenario: Cost of wrongly denying a good customer vs. accepting a bad loan
cost_analysis = {
    'wrongly_denied_good_customer': 1000,  # Lost revenue
    'bad_loan_default': 5000,               # Loss on principal + interest
}

# Easy Ensemble at threshold=0.3:
# - FP=500 wrongly denied → $500k loss
# - FN=87 bad loans → $435k loss
# - Total cost: $935k
```

**Question for stakeholder:** What threshold minimizes total cost?

### 7. **Cross-Validation (Prevent Overfitting Claims)**
- [ ] Use StratifiedKFold (k=5) on EACH resampling method
- [ ] Report mean ± std of metrics across folds
- **Why:** Single train-test split can be lucky/unlucky by chance

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_validate(model, X_train, y_train, cv=cv,
                        scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
```

**Expected:** Metrics should be stable (low std) → not just train-test luck

### 8. **Hyperparameter Tuning (Optional but impressive)**
- [ ] Tune Easy Ensemble: `n_estimators`, `max_depth`
- [ ] Tune Balanced RF: `max_depth`, `min_samples_split`
- **Tool:** GridSearchCV or RandomizedSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}
grid_search = GridSearchCV(EasyEnsembleClassifier(random_state=1), 
                           param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)
```

---

## 🚀 ADVANCED ANALYSIS (Next month)

### 9. **Learning Curves**
- [ ] Plot: training set size vs. model performance
- **Why:** Understand if you need more data or model is enough
- **Question:** Will performance plateau or keep improving with 10x more loans?

### 10. **Calibration Analysis**
- [ ] Check if predicted probabilities match actual probabilities
- [ ] Example: When model says "80% high risk," do 80% actually default?

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

prob_true, prob_pred = calibration_curve(y_test, y_proba_eec)
# If perfectly calibrated: prob_true ≈ prob_pred
```

**Why:** Important for setting business rules (e.g., "reject if >75% risk")

### 11. **Isolation Forest for Anomaly Detection**
- [ ] Identify unusual loan applications (potential fraud/errors)
- **Insight:** Maybe easy ensemble fails on rare patterns

```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.predict(X_test)  # -1 = anomaly
```

### 12. **SHAP or LIME Explainability**
- [ ] Explain individual predictions
- **Example:** "This loan is high-risk because: high interest_rate (+0.08 probability), low income (-0.05), recent delinquency (+0.10)"

```python
import shap
explainer = shap.Explainer(eec_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 📈 EXPECTED IMPROVEMENTS

| Analysis | Current | With Improvements |
|----------|---------|-------------------|
| **Best Model** | Easy Ensemble (82% acc) | Easy Ensemble (unchanged) |
| **Best Metric** | Recall 76% | Recall 85%+ (tuned threshold) |
| **Actionability** | Model comparison | Cost-benefit → business decision |
| **Trust Level** | Single test set | 5-fold CV ± std dev |
| **Interpretability** | "High recall" | "Top 5 features: X, Y, Z" + SHAP values |

---

## 💾 DELIVERABLES CHECKLIST

### Notebook 1: `credit_risk_resampling_IMPROVED.ipynb`
- [x] Initial EDA (already have)
- [x] 4 resampling methods (already have)
- [ ] **NEW:** Side-by-side confusion matrices
- [ ] **NEW:** ROC-AUC comparison
- [ ] **NEW:** 5-fold cross-validation results
- [ ] **NEW:** Summary: Why resampling alone struggles

### Notebook 2: `credit_risk_ensemble_IMPROVED.ipynb`
- [x] Balanced RF + Easy Ensemble (already have)
- [ ] **NEW:** Feature importance comparison chart
- [ ] **NEW:** Threshold optimization (F1, balanced accuracy, G-mean)
- [ ] **NEW:** Cost-benefit analysis at 3-4 thresholds
- [ ] **NEW:** Final recommendation with business context

### New File: `credit_risk_summary_report.md`
- [ ] 1-page executive summary
- [ ] Model comparison table (all 6 models, all metrics)
- [ ] Recommendation: "Use Easy Ensemble at threshold=0.30 because..."
- [ ] Business impact quantified in dollars/percentages

---

## 🎓 SKILLS DEMONSTRATED AFTER IMPROVEMENTS

**Current:**
- ✅ Data preprocessing (get_dummies, train-test split)
- ✅ Imbalanced learning (6 resampling/ensemble methods)
- ✅ Model evaluation (accuracy, precision, recall)

**After improvements:**
- ✅ **Advanced metrics** (F1, ROC-AUC, balanced accuracy, specificity)
- ✅ **Business context** (cost-benefit, threshold tuning)
- ✅ **Cross-validation** (preventing overfitting claims)
- ✅ **Visualizations** (ROC curves, confusion matrices, feature importance)
- ✅ **Hyperparameter tuning** (GridSearchCV)
- ✅ **Interpretability** (SHAP, feature importance)

---

## 📝 RECOMMENDED READING

1. **"Imbalanced Classification with Python"** by Jason Brownlee
   - Covers all 6 methods + more
   - Free on Machine Learning Mastery

2. **Scikit-learn Docs:**
   - https://scikit-learn.org/stable/modules/model_evaluation.html
   - ROC-AUC, precision-recall curves

3. **Why not Accuracy:**
   - https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

---

## 🔍 FINAL CHECKPOINT

**Before you call this project "done":**

- [ ] Can you explain why Easy Ensemble beats Naive Random? (ensemble > resampling)
- [ ] Can you explain the precision-recall trade-off in business terms?
- [ ] Can you recommend a specific threshold + justify it?
- [ ] Can you show which features matter most?
- [ ] Can you quantify the business impact (dollars)?

**If you can answer all 5:** You've mastered credit risk modeling! ✅
