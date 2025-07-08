# -*- coding: utf-8 -*-
"""Magic.ipynb - Final Clean Version for Training & Saving"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
import joblib

# Load data
cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "FM3Trans", 'fAlpha', "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)

# Convert class labels
df["class"] = (df["class"] == "g").astype(int)

# Plot each feature
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Split data
train_frac = 0.6
valid_frac = 0.2

train = df.sample(frac=train_frac, random_state=42)
valid = df.drop(train.index).sample(frac=valid_frac / (1 - train_frac), random_state=42)
test = df.drop(train.index).drop(valid.index)

# Preprocessing function
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    return X, y

# Preprocess datasets
X_train, y_train = scale_dataset(train, oversample=True)
X_valid, y_valid = scale_dataset(valid)
X_test, y_test = scale_dataset(test)

# Models – Evaluation
print("🔍 KNN")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("🔍 Naive Bayes")
nb_model = GaussianNB().fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("🔍 Logistic Regression")
lg_model = LogisticRegression().fit(X_train, y_train)
y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("🔍 SVM")
svm_model = SVC().fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Neural Network
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(X_train.shape[1], )),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid), verbose=0)

    return model, history

# Train multiple configs, choose best
least_val_loss = float('inf')
least_loss_model = None
epochs = 10

for num_nodes in [32]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.05, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"Training: nodes={num_nodes}, dropout={dropout_prob}, lr={lr}, batch={batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss, val_acc = model.evaluate(X_valid, y_valid)
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model

# Final evaluation
y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
print("🔍 Final Neural Network Performance:")
print(classification_report(y_test, y_pred))

# ✅ Save the model properly (Keras format)
least_loss_model.save("model.h5")

# ✅ Save the scaler
scaler = StandardScaler()
scaler.fit(train[cols[:-1]])
joblib.dump(scaler, "scaler.pkl")

print("✅ Keras model saved as model.h5")
print("✅ Scaler saved as scaler.pkl")
