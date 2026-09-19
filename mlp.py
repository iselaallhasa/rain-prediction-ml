import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("weather_cleaned_preprocessed.csv")

X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print(class_weights)
model = Sequential()

model.add(Dense(
    256,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)
early_stopping = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_mlp.keras",
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)
y_prob = model.predict(X_test).flatten()

threshold = 0.50

y_pred = (y_prob >= threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)
print(
    classification_report(
        y_test,
        y_pred
    )
)
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("MLP Confusion Matrix")
plt.show()
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend([
    "Train",
    "Validation"
])

plt.title("Accuracy")
plt.show()
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.legend([
    "Train",
    "Validation"
])

plt.title("Loss")
plt.show()
model.save("mlp_weather_model.keras")

print("MLP model saved successfully")
