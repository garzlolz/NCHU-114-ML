import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")

# 讀取資料
data = pd.read_csv("../week7_hw_data_processing/output/feature_select_by_myself.csv")
X = data.drop("y", axis=1)
y = data["y"]

print(f"資料形狀: {X.shape}")
print(f"特徵數量: {X.shape[1]}")
print(f"\n目標變數分布:\n{y.value_counts()}")

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1000
)

print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 1. SVM (GridSearchCV)
print("\n" + "="*50)
print("SVM GridSearchCV 訓練")
print("="*50)

y_train_svm = y_train.replace({0: -1, 1: 1})
y_test_svm = y_test.replace({0: -1, 1: 1})

param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 5, 10],
    "gamma": ["scale", 0.01, 0.1],
}

gs = GridSearchCV(
    SVC(probability=True, random_state=1000),
    param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2,
)

start_svm = time.time()
gs.fit(X_train, y_train_svm)
train_svm = time.time() - start_svm

print(f"\nGridSearchCV 完成, 耗時: {train_svm:.3f} 秒")
print(f"最佳 SVM 參數: {gs.best_params_}")
print(f"最佳交叉驗證分數: {gs.best_score_:.4f}")

best_svm = gs.best_estimator_
y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test_svm, y_pred_svm)
prec_svm = precision_score(y_test_svm, y_pred_svm)
rec_svm = recall_score(y_test_svm, y_pred_svm)
f1_svm = f1_score(y_test_svm, y_pred_svm)
auc_svm = roc_auc_score(y_test_svm, y_proba_svm)

print(f"\nSVM 效能:")
print(f"訓練時間: {train_svm:.3f} 秒")
print(f"Accuracy: {acc_svm:.4f}")
print(f"Precision: {prec_svm:.4f}")
print(f"Recall: {rec_svm:.4f}")
print(f"F1-Score: {f1_svm:.4f}")
print(f"AUC: {auc_svm:.4f}")

# 2. sklearn MLPClassifier
print("\n" + "="*50)
print("sklearn MLPClassifier 訓練")
print("="*50)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=1000,
    verbose=False
)

start_mlp = time.time()
mlp.fit(X_train, y_train)
train_mlp = time.time() - start_mlp

y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)[:, 1]

acc_mlp = accuracy_score(y_test, y_pred_mlp)
prec_mlp = precision_score(y_test, y_pred_mlp)
rec_mlp = recall_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)
auc_mlp = roc_auc_score(y_test, y_proba_mlp)

print(f"\nsklearn MLP 效能 (架構: (64, 32)):")
print(f"訓練時間: {train_mlp:.3f} 秒")
print(f"Accuracy: {acc_mlp:.4f}")
print(f"Precision: {prec_mlp:.4f}")
print(f"Recall: {rec_mlp:.4f}")
print(f"F1-Score: {f1_mlp:.4f}")
print(f"AUC: {auc_mlp:.4f}")

# 3. TensorFlow Keras
print("\n" + "="*50)
print("TensorFlow Keras Neural Network 訓練")
print("="*50)

visible = Input(shape=(X_train.shape[1],))
hidden1 = Dense(64, activation='tanh')(visible)
hidden2 = Dense(32, activation='tanh')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)

model = Model(inputs=visible, outputs=output)
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

print("\n模型架構:")
model.summary()

start_tf = time.time()
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
train_tf = time.time() - start_tf

y_proba_tf = model.predict(X_test, verbose=0).flatten()
y_pred_tf = (y_proba_tf > 0.5).astype(int)

acc_tf = accuracy_score(y_test, y_pred_tf)
prec_tf = precision_score(y_test, y_pred_tf)
rec_tf = recall_score(y_test, y_pred_tf)
f1_tf = f1_score(y_test, y_pred_tf)
auc_tf = roc_auc_score(y_test, y_proba_tf)

print(f"\nTensorFlow Keras NN 效能 (架構: (64, 32)):")
print(f"訓練時間: {train_tf:.3f} 秒")
print(f"Accuracy: {acc_tf:.4f}")
print(f"Precision: {prec_tf:.4f}")
print(f"Recall: {rec_tf:.4f}")
print(f"F1-Score: {f1_tf:.4f}")
print(f"AUC: {auc_tf:.4f}")

# 模型比較表格
print("\n" + "="*50)
print("三種模型效能比較")
print("="*50)

comparison = pd.DataFrame({
    "模型": ["SVM (GridSearchCV)", "sklearn MLP (64,32)", "TensorFlow Keras (64,32)"],
    "訓練時間 (秒)": [f"{train_svm:.3f}", f"{train_mlp:.3f}", f"{train_tf:.3f}"],
    "Accuracy": [f"{acc_svm:.4f}", f"{acc_mlp:.4f}", f"{acc_tf:.4f}"],
    "Precision": [f"{prec_svm:.4f}", f"{prec_mlp:.4f}", f"{prec_tf:.4f}"],
    "Recall": [f"{rec_svm:.4f}", f"{rec_mlp:.4f}", f"{rec_tf:.4f}"],
    "F1-Score": [f"{f1_svm:.4f}", f"{f1_mlp:.4f}", f"{f1_tf:.4f}"],
    "AUC": [f"{auc_svm:.4f}", f"{auc_mlp:.4f}", f"{auc_tf:.4f}"]
})

print("\n")
print(comparison.to_string(index=False))

best_models = {
    "Accuracy": comparison.loc[comparison["Accuracy"].astype(float).idxmax(), "模型"],
    "Precision": comparison.loc[comparison["Precision"].astype(float).idxmax(), "模型"],
    "Recall": comparison.loc[comparison["Recall"].astype(float).idxmax(), "模型"],
    "F1-Score": comparison.loc[comparison["F1-Score"].astype(float).idxmax(), "模型"],
    "AUC": comparison.loc[comparison["AUC"].astype(float).idxmax(), "模型"],
}

print("\n各指標最佳模型:")
for metric, model_name in best_models.items():
    print(f"  {metric}: {model_name}")

# TensorFlow 訓練歷史視覺化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="訓練準確率", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="驗證準確率", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("TensorFlow Keras 訓練準確率")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="訓練損失", linewidth=2)
axes[1].plot(history.history["val_loss"], label="驗證損失", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("TensorFlow Keras 訓練損失")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n結論:")
print("- 架構: 22特徵用 (64,32) 兩層隱藏層")
print("- SVM: Accuracy/Precision最佳 (1613s)")
print("- sklearn MLP: Recall最佳+最快 (24s)") 
print("- TensorFlow Keras: AUC最佳 (249s)")
