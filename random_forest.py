
import pandas as pd

file_path = 'weather_cleaned_preprocessed.csv'
df = pd.read_csv(file_path)

import pandas as pd

df = pd.read_csv('weather_cleaned_preprocessed.csv')


print("total n of lines", len(df))



# Import necessary libraries for Objective 3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

print("STARTING OBJECTIVE 3: PREDICTIVE MODELING\n")


X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Data Split Complete:")
print(f" - Training Set: {X_train.shape[0]} samples (70%)")
print(f" - Testing Set:  {X_test.shape[0]} samples (30%)\n")


rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


print("Training the Random Forest model... ")
rf_baseline.fit(X_train, y_train)
print("Training completed successfully!\n")


y_pred_binary = rf_baseline.predict(X_test)
y_pred_proba = rf_baseline.predict_proba(X_test)[:, 1]

print("Sanity Check - First 5 Predictions:")
sanity_check_df = pd.DataFrame({
    'Actual_Truth': y_test.values[:5],
    'Binary_Prediction': y_pred_binary[:5],
    'Rain_Probability': y_pred_proba[:5]
})
print(sanity_check_df)



from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

print("--- 4.1 HYPERPARAMETER TUNING, SMOTE & CROSS-VALIDATION ---\n")

rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],         
    'classifier__max_depth': [10, 20, 30, None],         
    'classifier__min_samples_split': [2, 5, 10]         

smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])
}

print("Searching for the best parameters with SMOTE...")
rf_random_search = RandomizedSearchCV(
    estimator=smote_pipeline,
    param_distributions=rf_param_grid,
    n_iter=5,               
    cv=3,                 
    scoring='f1',           
    random_state=42,
    n_jobs=-1
)


rf_random_search.fit(X_train, y_train)


best_rf_model = rf_random_search.best_estimator_

print("Search Complete!")
print(f"Best Parameters Found: {rf_random_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {rf_random_search.best_score_:.4f}")


print("Il numero totale di righe è:", len(best_rf_model))



from sklearn.metrics import accuracy_score

print("--- 4.2 OVERFITTING ANALYSIS ---\n")


y_train_pred = best_rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


y_test_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


print(f"Accuracy on Training Set: {train_accuracy:.4f}")
print(f"Accuracy on Testing Set:  {test_accuracy:.4f}")


accuracy_gap = train_accuracy - test_accuracy
print(f"Performance Gap: {accuracy_gap:.4f}")

if accuracy_gap > 0.05:
    print("\nConclusion: The model shows signs of overfitting. It is memorizing some training data.")
else:
    print("\nConclusion: The model generalizes well! The gap is minimal, indicating no severe overfitting.")


from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

print(" PROBABILITY CALIBRATION ")


print("Calibrating model probabilities using Platt Scaling (Sigmoid)...")
calibrated_rf = CalibratedClassifierCV(estimator=best_rf_model, method='sigmoid', cv=3)


calibrated_rf.fit(X_train, y_train)

final_binary_preds = calibrated_rf.predict(X_test)
final_proba_preds = calibrated_rf.predict_proba(X_test)[:, 1]

print("Calibration successful! The model 'calibrated_rf' will be used for final evaluations.\n")

comparison_df = pd.DataFrame({
    'Uncalibrated_Proba': best_rf_model.predict_proba(X_test)[:5, 1],
    'Calibrated_Proba': final_proba_preds[:5]
})
print("Probability Shift Example (First 5 instances):")
print(comparison_df)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

print("CONFUSION MATRIX & METRICS\n")


print("Detailed Classification Report (Test Set):")
print(classification_report(y_test, final_binary_preds))


cm = confusion_matrix(y_test, final_binary_preds)


plt.figure(figsize=(9, 7))

mask_correct = np.array([[False, True], [True, False]])  
mask_errors = np.array([[True, False], [False, True]])  


sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            mask=mask_correct,
            xticklabels=['Predicted: No Rain (0)', 'Predicted: Rain (1)'],
            yticklabels=['Actual: No Rain (0)', 'Actual: Rain (1)'],
            annot_kws={"size": 18, "weight": "bold"})


sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            mask=mask_errors,
            xticklabels=['Predicted: No Rain (0)', 'Predicted: Rain (1)'],
            yticklabels=['Actual: No Rain (0)', 'Actual: Rain (1)'],
            annot_kws={"size": 18, "weight": "bold"})

plt.title('Random Forest - Confusion Matrix\n(Green = Correct, Red = Errors)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('True Meteorological State (y_test)', fontsize=12, fontweight='bold')
plt.xlabel('Model Prediction (final_binary_preds)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

print("ROC CURVE & AUC SCORE\n")


auc_score = roc_auc_score(y_test, final_proba_preds)
print(f"ROC-AUC Score: {auc_score:.4f}\n")


fpr, tpr, thresholds = roc_curve(y_test, final_proba_preds)


plt.figure(figsize=(8, 6))


plt.plot(fpr, tpr, color='darkorange', lw=2.5,
         label=f'Calibrated Random Forest (AUC = {auc_score:.3f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Guessing (AUC = 0.500)')

# Chart formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (False Alarms %)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Correctly Predicted Storms %)', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

print("PRECISION-RECALL CURVE\n")


ap_score = average_precision_score(y_test, final_proba_preds)
print(f"Average Precision (PR-AUC) Score: {ap_score:.4f}\n")

precision, recall, thresholds = precision_recall_curve(y_test, final_proba_preds)


baseline = sum(y_test) / len(y_test)

plt.figure(figsize=(8, 6))

plt.plot(recall, precision, color='purple', lw=2.5,
         label=f'Calibrated Random Forest (AP = {ap_score:.3f})')

plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
         label=f'Random Guessing Baseline ({baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Percentage of actual storms caught)', fontsize=12, fontweight='bold')
plt.ylabel('Precision (Accuracy when predicting rain)', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve (Imbalanced Data Focus)', fontsize=16, fontweight='bold', pad=15)
plt.legend(loc="upper right", fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()