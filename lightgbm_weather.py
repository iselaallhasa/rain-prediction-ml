import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import lightgbm as lgb

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
df = pd.read_csv("weather_cleaned_preprocessed.csv")

print("Dataset Shape:", df.shape)

X = df.drop("RainTomorrow", axis=1)

y = df["RainTomorrow"]

print("\nTarget Distribution:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42,
    n_jobs=-1
)
param_grid = {

    "n_estimators": [300, 500, 800],

    "learning_rate": [0.01, 0.03, 0.05],

    "max_depth": [5, 7, 10],

    "num_leaves": [31, 50, 80],

    "subsample": [0.8, 0.9, 1.0],

    "colsample_bytree": [0.8, 0.9, 1.0],

    "min_child_samples": [20, 50, 100]
}
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring='f1',
    cv=cv,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

random_search.fit(
    X_train_smote,
    y_train_smote
)

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest CV Score:")
print(random_search.best_score_)
best_model = random_search.best_estimator_

best_model.fit(
    X_train_smote,
    y_train_smote
)
y_prob = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(
    0.20,
    0.81,
    0.01
)

best_threshold = 0.50
best_f1 = 0

for threshold in thresholds:

    preds = (y_prob >= threshold).astype(int)

    score = f1_score(
        y_test,
        preds
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best F1:", best_f1)
y_pred = (
    y_prob >= best_threshold
).astype(int)
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)
from sklearn.metrics import roc_curve



fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"LightGBM (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    'k--',
    label='Random Classifier'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - LightGBM")

plt.legend()

plt.figtext(
    0.5,
    -0.08,
    """
Blue curve:
Shows the performance of the LightGBM model.

Dashed diagonal line:
Represents a random classifier.

The closer the curve is to the top-left corner,
the better the model separates Rain and No Rain classes.
""",
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

print("\RESULTS ")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

print("ROC-AUC  :", roc_auc)
print(
    classification_report(
        y_test,
        y_pred
    )
)


cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Rain', 'Rain'],
    yticklabels=['No Rain', 'Rain']
)

plt.title("LightGBM Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.figtext(
    0.5,
    -0.08,
    f"""
TN (True Negative) = {TN}
→ Correctly predicted No Rain

FP (False Positive) = {FP}
→ Predicted Rain but actually No Rain

FN (False Negative) = {FN}
→ Predicted No Rain but actually Rain

TP (True Positive) = {TP}
→ Correctly predicted Rain
""",
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()


precision_vals, recall_vals, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_vals,
    precision_vals,
    linewidth=2
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve - LightGBM")

plt.figtext(
    0.5,
    -0.08,

    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()
importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(
    importance_df.head(20)
)
plt.figure(figsize=(10,8))

sns.barplot(
    data=importance_df.head(20),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 20 Feature Importance"
)

plt.show()
import shap

explainer = shap.TreeExplainer(
    best_model
)

shap_values = explainer.shap_values(
    X_test.sample(
        1000,
        random_state=42
    )
)

shap.summary_plot(
    shap_values,
    X_test.sample(
        1000,
        random_state=42
    )
)
