# Code Review Checklist for Credit Risk Notebooks

## 🔍 `credit_risk_resampling.ipynb`

### **Section 1: Data Loading & Exploration**

```python
# ✅ CHECK: Do you have code like this?
df = pd.read_csv('LoanStats_2019Q1.csv')
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df['loan_status'].value_counts())  # Shows class imbalance
```

**Issues to look for:**
- [ ] Are there missing values? How do you handle them?
  - Good: Drop rows, impute, or flag as separate category
  - Bad: Ignore and proceed
- [ ] Do you check data types? (Should be numeric after encoding)
- [ ] Do you visualize the class imbalance?
  - `df['loan_status'].value_counts().plot(kind='bar')`

---

### **Section 2: Feature Engineering**

```python
# ✅ GOOD PATTERN:
X = pd.get_dummies(df.drop(columns='loan_status'))
y = df['loan_status']

print(f"Feature shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
```

**⚠️ CRITICAL ISSUES:**

| Issue | Bad Code | Good Code | Why |
|-------|----------|-----------|-----|
| **Data leakage** | `X, y = df.iloc[:]` before split | `y = df['loan_status']` then split | One-hot encoding can be different train vs test |
| **Dummy variables** | `X = pd.get_dummies(df)` (includes target) | `X = pd.get_dummies(df.drop(columns='loan_status'))` | Target should not be in features |
| **Unseen categories** | No handling | Use `drop_first=True` or handle in pipeline | Prevents errors on new test categories |

```python
# ⚠️ BAD:
X = df.drop(columns='loan_status')
X = pd.get_dummies(X)  # Only encodes train categories
# Now if test has a new category → error or silent bug

# ✅ GOOD:
X = pd.get_dummies(X, drop_first=True)  # Remove one to avoid multicollinearity
```

---

### **Section 3: Train-Test Split**

```python
# ✅ PERFECT:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
print(f"Train size: {X_train.shape}")
print(f"Test distribution:\n{y_test.value_counts()}")
print(f"Train distribution:\n{y_train.value_counts()}")
```

**⚠️ CRITICAL ISSUE — Data Leakage:**

```python
# ❌ WRONG (DO NOT DO THIS):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Then resample BEFORE split:
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)  # ← LEAKAGE! Test set influenced

# ✅ CORRECT (DO THIS):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# Then resample AFTER split:
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # ← Only train!
```

**Checklist:**
- [ ] Do you use `stratify=y`? (Ensures same class distribution train vs test)
- [ ] Do you print class distributions AFTER split?
- [ ] Do you resample AFTER splitting?
- [ ] Do you use `random_state=1` for reproducibility?

---

### **Section 4: Resampling Methods**

#### **4a. Naive Random Oversampling**

```python
# ✅ CORRECT:
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

print(f"Original train distribution:\n{y_train.value_counts()}")
print(f"Resampled train distribution:\n{y_ros.value_counts()}")
# Should be 50/50 or close
```

**Issues to check:**
- [ ] Do you print before/after distributions?
- [ ] Do you apply to X_train/y_train only (not test)?
- [ ] Do you preserve random_state for reproducibility?

---

#### **4b. SMOTE Oversampling**

```python
# ✅ CORRECT:
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"Resampled shape: {X_smote.shape}")
# Check that new samples are interpolated, not copies
```

**Issues to check:**
- [ ] Do you set `k_neighbors=5`? (Creates interpolation between 5 nearest neighbors)
- [ ] Do you check that output shape is reasonable?
- [ ] Do you understand it creates *synthetic* samples (not duplicates)?

**Quick test:**
```python
# Verify SMOTE didn't just copy rows
print(f"Unique original rows: {len(X_train.drop_duplicates())}")
print(f"Unique resampled rows: {len(X_smote.drop_duplicates())}")
# Resampled should be ALL unique (synthetic != duplicates)
```

---

#### **4c. Cluster Centroid Undersampling**

```python
# ✅ CORRECT:
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1, n_clusters=<minority_class_count>)
X_cc, y_cc = cc.fit_resample(X_train, y_train)

print(f"Original train shape: {X_train.shape}")
print(f"Undersampled shape: {X_cc.shape}")
# Shape[0] should be ~2x minority class (50/50 split)
```

**Issues to check:**
- [ ] Do you set `n_clusters` = minority class count? (For perfect 50/50 balance)
- [ ] Are you aware this LOSES data from majority class?
- [ ] Do you understand centroids may not be real loan applications?

---

#### **4d. SMOTEENN (Combination Sampling)**

```python
# ✅ CORRECT:
from imblearn.combine import SMOTEENN

smoteenn = SMOTEENN(random_state=1)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train, y_train)

print(f"After SMOTEENN shape: {X_smoteenn.shape}")
print(f"Class distribution:\n{pd.Series(y_smoteenn).value_counts()}")
# Should be balanced AND cleaned of noisy samples
```

**Issues to check:**
- [ ] Do you understand it does SMOTE first, then removes noisy samples?
- [ ] Do you check final class distribution is balanced?
- [ ] This should have best of both worlds: synthetic + clean boundaries

---

### **Section 5: Logistic Regression Training**

```python
# ✅ CORRECT:
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1, max_iter=1000)
lr.fit(X_train_resampled, y_train_resampled)  # ← Use RESAMPLED data

# ⚠️ Important: Use original (non-resampled) test set
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]  # Probability of high-risk class
```

**Issues to check:**
- [ ] Do you set `max_iter=1000`? (Avoids convergence warning)
- [ ] Do you train on resampled data, test on ORIGINAL test set?
- [ ] Do you extract `predict_proba()` for threshold optimization later?
- [ ] Do you store BOTH `y_pred` (0/1) AND `y_proba` (probabilities)?

---

### **Section 6: Evaluation**

```python
# ✅ COMPREHENSIVE EVALUATION:
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)

# Basic metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nConfusion Matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Precision: {tp/(tp+fp):.3f}")
print(f"Recall: {tp/(tp+fn):.3f}")
print(f"Specificity: {tn/(tn+fp):.3f}")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

**⚠️ ISSUES TO CHECK:**

| Metric | What to Check | Why |
|--------|---------------|-----|
| **Accuracy** | Should be >50% but don't trust it for imbalanced data | Can be high by predicting majority class always |
| **Balanced Accuracy** | Should match your final metric | This accounts for class imbalance |
| **ROC-AUC** | Should be >0.5 (random is 0.5) | BEST metric for imbalanced data |
| **Precision** | Low (0.01-0.03) is OK if recall is high | Means high false alarm rate |
| **Recall** | Should be high (>0.7) to catch defaults | Missing defaults is expensive |
| **F1 Score** | Harmonic mean of precision & recall | Balances the trade-off |

**Common mistakes:**
```python
# ❌ WRONG:
print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
# This only works for binary classification and may throw errors

# ✅ CORRECT:
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
```

---

### **Summary for Resampling Notebook**

**Ideal structure:**
```
1. Load data
   ↓
2. Exploratory analysis (show imbalance)
   ↓
3. Feature engineering (one-hot encoding)
   ↓
4. Train-test split (stratified, NO data leakage)
   ↓
5. For each of 4 resampling methods:
   ├─ Resample X_train/y_train only
   ├─ Train Logistic Regression
   ├─ Predict on ORIGINAL X_test
   ├─ Calculate metrics (accuracy, balanced_accuracy, ROC-AUC, F1, precision, recall)
   └─ Print confusion matrix
   ↓
6. Summary table comparing all 4 methods
```

---

## 🔍 `credit_risk_ensemble.ipynb`

### **Section 1: Balanced Random Forest**

```python
# ✅ CORRECT:
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=1,
    max_depth=None,
    min_samples_split=2
)
brf.fit(X_train, y_train)  # ← No resampling needed! Built-in balancing
y_pred_brf = brf.predict(X_test)
y_proba_brf = brf.predict_proba(X_test)[:, 1]

# Feature importance (unique to tree models)
feature_importance = pd.Series(
    brf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print(feature_importance.head(10))
```

**Issues to check:**
- [ ] Do you use raw X_train (no resampling before fit)?
- [ ] Do you set `n_estimators=100` (or similar)?
- [ ] Do you extract feature importance?
- [ ] Do you recognize it balances by undersampling each tree's bootstrap?

---

### **Section 2: Easy Ensemble**

```python
# ✅ CORRECT:
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(
    n_estimators=100,
    random_state=1,
    estimator=DecisionTreeClassifier(max_depth=None)
)
eec.fit(X_train, y_train)
y_pred_eec = eec.predict(X_test)
y_proba_eec = eec.predict_proba(X_test)[:, 1]
```

**Issues to check:**
- [ ] Do you set `n_estimators` (number of weak learners)?
- [ ] Do you understand it trains multiple classifiers on balanced subsets?
- [ ] Do you store predict_proba for later threshold tuning?

---

### **Section 3: Comparison**

```python
# ✅ BEST PRACTICE:
results = {
    'Balanced RF': {
        'accuracy': balanced_accuracy_score(y_test, y_pred_brf),
        'precision': tp_brf / (tp_brf + fp_brf),
        'recall': tp_brf / (tp_brf + fn_brf),
        'f1': f1_score(y_test, y_pred_brf),
        'roc_auc': roc_auc_score(y_test, y_proba_brf)
    },
    'Easy Ensemble': {
        # ... same metrics
    }
}

results_df = pd.DataFrame(results).T
print(results_df)
```

**Issues to check:**
- [ ] Do you compare using same metrics as resampling methods?
- [ ] Do you include ROC-AUC in comparison?
- [ ] Do you create a summary table combining ALL 6 methods?

---

## 📋 FINAL CHECKLIST

### **Before submitting resampling notebook:**

- [ ] ✅ Data loading with exploration
- [ ] ✅ Stratified train-test split
- [ ] ✅ No data leakage (resample AFTER split)
- [ ] ✅ 4 resampling methods applied correctly
- [ ] ✅ Logistic Regression trained on resampled data
- [ ] ✅ Evaluated on ORIGINAL test set
- [ ] ✅ Metrics: accuracy, balanced_accuracy, ROC-AUC, F1, precision, recall
- [ ] ✅ Confusion matrices printed for each method
- [ ] ✅ Summary table comparing all 4 methods
- [ ] ✅ Both `y_pred` and `y_proba` stored

### **Before submitting ensemble notebook:**

- [ ] ✅ Balanced Random Forest trained
- [ ] ✅ Easy Ensemble trained
- [ ] ✅ Feature importance extracted + visualized
- [ ] ✅ Same metrics as resampling notebook
- [ ] ✅ All 6 methods (resampling + ensemble) in single comparison table
- [ ] ✅ ROC-AUC curves plotted
- [ ] ✅ Confusion matrices side-by-side (optional but impressive)
- [ ] ✅ Recommendation: "Easy Ensemble is best because..."

---

## 🎓 If you check ✅ on ALL of these, your project is PUBLICATION-READY!
