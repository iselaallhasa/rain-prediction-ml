import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('weather_cleaned_preprocessed.csv')

print(df.head())
print(df.isnull().sum())
print(df['RainTomorrow'].isnull().sum())
print(df['RainTomorrow'].value_counts(dropna=False))

df = df.dropna(subset=['RainTomorrow']).copy()

print(df['RainTomorrow'].value_counts())

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#SVM Model Part

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#Applying Smote for the SVM part
print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())


pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_smote)

X_test_pca = pca.transform(X_test)

linear_svm = LinearSVC(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=5000
)


svm_model = CalibratedClassifierCV(linear_svm)


svm_cv_scores = cross_val_score(
    svm_model,
    X_train_pca,
    y_train_smote,
    cv=skf,
    scoring='accuracy'
)

print('Cross Validation Scores:', svm_cv_scores)

print('Mean CV Accuracy:', svm_cv_scores.mean())

svm_model.fit(
    X_train_pca,
    y_train_smote
)

y_prob_svm = svm_model.predict_proba(X_test_pca)[:, 1]

threshold = 0.45

y_pred_svm = (y_prob_svm >= threshold).astype(int)


svm_accuracy = accuracy_score(y_test, y_pred_svm)

svm_precision = precision_score(y_test, y_pred_svm)

svm_recall = recall_score(y_test, y_pred_svm)

svm_f1 = f1_score(y_test, y_pred_svm)

print('SVM Accuracy:', svm_accuracy)

print('SVM Precision:', svm_precision)

print('SVM Recall:', svm_recall)

print('SVM F1 Score:', svm_f1)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred_svm
).ravel()

plt.figure(figsize=(7,6))

sns.heatmap(
    [[tn, fp],
     [fn, tp]],
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    'SVM Confusion Matrix'
)

plt.xlabel(
    'Predicted Class'
)

plt.ylabel(
    'Actual Class'
)

plt.figtext(
    0.5,
    -0.12,
    f'TN={tn}: Correct No Rain predictions\n'
    f'FP={fp}: False Rain alarms\n'
    f'FN={fn}: Missed Rain events\n'
    f'TP={tp}: Correct Rain predictions',
    ha='center'
)

plt.show()


svm_train_accuracy = svm_model.score(
    X_train_pca,
    y_train_smote
)

svm_test_accuracy = svm_model.score(
    X_test_pca,
    y_test
)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print("SVM Train Accuracy:", svm_train_accuracy)

print("SVM Test Accuracy:", svm_test_accuracy)

print(classification_report(
    y_test,
    y_pred_svm
))


coef_importance = np.abs(
    svm_model.calibrated_classifiers_[0].estimator.coef_[0]
)

feature_names = [
    f"PC{i+1}"
    for i in range(len(coef_importance))
]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coef_importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(15)
)

plt.title(
    'SVM Feature Importance (Principal Components)'
)

plt.figtext(
    0.5,
    -0.05,
    'Higher values indicate components that contribute more strongly\n'
    'to the decision boundary separating Rain and No Rain.',
    ha='center'
)

plt.show()

import shap

explainer = shap.LinearExplainer(
    svm_model.calibrated_classifiers_[0].estimator,
    X_train_pca
)

shap_values = explainer.shap_values(
    X_test_pca[:1000]
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_pca[:1000],
    show=False
)

plt.title(
    'SVM SHAP Summary Plot'
)

plt.figtext(
    0.5,
    -0.05,
    'Red points increase Rain prediction.\n'
    'Blue points decrease Rain prediction.\n'
    'Features at the top have the largest impact.',
    ha='center'
)

plt.show()


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

svm_auc = roc_auc_score(
    y_test,
    y_prob_svm
)

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob_svm
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'SVM ROC Curve (AUC = {svm_auc:.3f})'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2,
    label='Random Classifier Baseline'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.grid(True)

plt.figtext(
    0.5,
    -0.05,
    'Solid blue curve = SVM performance.\n'
    'Dashed diagonal line = random classifier baseline.\n'
    'Curves closer to the top-left corner indicate better discrimination capability.',
    ha='center',
    fontsize=10
)

plt.show()




precision_vals, recall_vals, _ = precision_recall_curve(
    y_test,
    y_prob_svm
)

ap_score = average_precision_score(
    y_test,
    y_prob_svm
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_vals,
    precision_vals,
    linewidth=2,
    label=f'SVM PR Curve (AP = {ap_score:.3f})'
)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('SVM Precision-Recall Curve')
plt.legend()
plt.grid(True)

plt.figtext(
    0.5,
    -0.05,
    'Blue curve shows the trade-off between Precision and Recall.\n'
    'A curve closer to the top-right corner indicates stronger performance.\n'
    'Precision-Recall curves are especially important for imbalanced datasets such as Rain Prediction.',
    ha='center',
    fontsize=10
)

plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


X_train_gru = np.array(X_train)
X_test_gru = np.array(X_test)
y_train_gru = np.array(y_train)
y_test_gru = np.array(y_test)

classes = np.unique(y_train_gru)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_gru
)

class_weights = dict(zip(classes, weights))

print("Class Weights:")
print(class_weights)


X_train_gru = X_train_gru.reshape(
    X_train_gru.shape[0],
    1,
    X_train_gru.shape[1]
)

X_test_gru = X_test_gru.reshape(
    X_test_gru.shape[0],
    1,
    X_test_gru.shape[1]
)


def build_gru_model(units=64, dropout_rate=0.3):

    model = Sequential()

    model.add(GRU(
        units=units,
        input_shape=(1, X_train_gru.shape[2]),
        return_sequences=False
    ))

    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



gru_results = []

units_list = [32, 64]
dropout_list = [0.2, 0.3]
batch_sizes = [32]
epochs_list = [10]


for units in units_list:
    for dropout in dropout_list:
        for batch_size in batch_sizes:
            for epochs in epochs_list:

                print(f"Training GRU: units={units}, dropout={dropout}")

                model = build_gru_model(units, dropout)

                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )

                history = model.fit(
                     X_train_gru,
                     y_train_gru,
                     validation_split=0.2,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[early_stopping],
                     class_weight=class_weights,
                     verbose=0
                )

                best_val_acc = max(history.history['val_accuracy'])

                gru_results.append({
                    'units': units,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'val_accuracy': best_val_acc
                })


gru_results_df = pd.DataFrame(gru_results)
gru_results_df = gru_results_df.sort_values(by='val_accuracy', ascending=False)

print(gru_results_df)

best = gru_results_df.iloc[0]


final_gru_model = build_gru_model(
    units=int(best['units']),
    dropout_rate=float(best['dropout'])
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


history = final_gru_model.fit(
    X_train_gru,
    y_train_gru,
    validation_split=0.2,
    epochs=int(best['epochs']),
    batch_size=int(best['batch_size']),
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)



gru_loss, gru_acc = final_gru_model.evaluate(X_test_gru, y_test_gru, verbose=0)

y_pred_prob = final_gru_model.predict(X_test_gru)
y_pred_gru = (y_pred_prob > 0.5).astype(int)

gru_accuracy = accuracy_score(y_test_gru, y_pred_gru)
gru_precision = precision_score(y_test_gru, y_pred_gru)
gru_recall = recall_score(y_test_gru, y_pred_gru)
gru_f1 = f1_score(y_test_gru, y_pred_gru)

print("GRU Accuracy:", gru_accuracy)
print("GRU Precision:", gru_precision)
print("GRU Recall:", gru_recall)
print("GRU F1:", gru_f1)


print(classification_report(
    y_test_gru,
    y_pred_gru
))

plt.show()


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

gru_auc = roc_auc_score(
    y_test_gru,
    y_pred_prob
)

fpr_gru, tpr_gru, _ = roc_curve(
    y_test_gru,
    y_pred_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr_gru,
    tpr_gru,
    linewidth=2,
    label=f'GRU ROC Curve (AUC = {gru_auc:.3f})'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2,
    label='Random Classifier Baseline'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GRU ROC Curve')
plt.legend()
plt.grid(True)

plt.figtext(
    0.5,
    -0.05,
    'Solid blue curve = GRU model performance.\n'
    'Dashed diagonal line = random classifier baseline.\n'
    'Higher curves indicate better separation between Rain and No Rain classes.',
    ha='center',
    fontsize=10
)

plt.show()



precision_gru, recall_gru, _ = precision_recall_curve(
    y_test_gru,
    y_pred_prob
)

ap_gru = average_precision_score(
    y_test_gru,
    y_pred_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_gru,
    precision_gru,
    linewidth=2,
    label=f'GRU PR Curve (AP = {ap_gru:.3f})'
)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('GRU Precision-Recall Curve')
plt.legend()
plt.grid(True)

plt.figtext(
    0.5,
    -0.05,
    'The Precision-Recall curve highlights model performance on the minority Rain class.\n'
    'Curves closer to the upper-right corner indicate better Rain detection capability.',
    ha='center',
    fontsize=10
)

plt.show()
tn, fp, fn, tp = confusion_matrix(
    y_test_gru,
    y_pred_gru
).ravel()

plt.figure(figsize=(7,6))

sns.heatmap(
    [[tn, fp],
     [fn, tp]],
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title(
    'GRU Confusion Matrix'
)

plt.xlabel(
    'Predicted Class'
)

plt.ylabel(
    'Actual Class'
)

plt.figtext(
    0.5,
    -0.12,
    f'TN={tn}: Correct No Rain predictions\n'
    f'FP={fp}: False Rain alarms\n'
    f'FN={fn}: Missed Rain events\n'
    f'TP={tp}: Correct Rain predictions',
    ha='center'
)

plt.show()


comparison_df = pd.DataFrame({
    'Model': ['SVM', 'GRU'],
    'Accuracy': [svm_accuracy, gru_accuracy],
    'Precision': [svm_precision, gru_precision],
    'Recall': [svm_recall, gru_recall],
    'F1 Score': [svm_f1, gru_f1]
})

print(comparison_df)


comparison_df.plot(
    x='Model',
    kind='bar',
    figsize=(10,6)
)

plt.title('SVM vs GRU Performance Comparison')
plt.show()


joblib.dump(svm_model, 'svm_model.pkl')
print("SVM saved")

final_gru_model.save('gru_model.h5')
print("GRU saved")


print("""
DONE

Models:
- SVM (Grid + Random Search)
- GRU Deep Learning

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
""")