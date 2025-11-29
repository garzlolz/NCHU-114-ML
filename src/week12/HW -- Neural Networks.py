# %% [markdown]
# # HW - Neural Network 與 SVM 比較
#
#     姓名: 施宏勲
#     學號: 5114029040
#
# ## 題目: 使用 sklearn 和 tensorflow 的 neural network 解決銀行定存問題，並與 SVM 做比較

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 設定中文字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 讀取前處理後的資料（已標準化）
data = pd.read_csv("../week7_hw_data_processing/output/feature_select_by_myself.csv")
X = data.drop("y", axis=1)
y = data["y"]

print(f"資料形狀: {X.shape}")
print(f"特徵數量: {X.shape[1]}")
print(f"\n目標變數分布:\n{y.value_counts()}")

# %% [markdown]
# ### 分割資料成訓練集和測試集

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1000
)

print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# %% [markdown]
# ### 1. SVM (GridSearchCV)

# %%
print("\n" + "=" * 50)
print("SVM GridSearchCV 訓練")
print("=" * 50)

# SVM 需要將 y 轉換為 -1 和 1
y_train_svm = y_train.replace({0: -1, 1: 1})
y_test_svm = y_test.replace({0: -1, 1: 1})

print(f"\nSVM 使用的標籤: {y_train_svm.unique()}")

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

# 使用最佳模型預測
best_svm = gs.best_estimator_
y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

# 計算指標時需要使用 y_test_svm
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

# %% [markdown]
# ### 2. sklearn MLPClassifier (Neural Network)
#
# 架構: 兩層隱藏層 (64, 32)，適合 22 個特徵的中等規模資料

# %%
print("\n" + "=" * 50)
print("sklearn MLPClassifier 訓練")
print("=" * 50)

# 使用 (64, 32) 架構：兩層隱藏層
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="tanh",
    solver="sgd",
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=1000,
    verbose=False,
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

# %% [markdown]
# ### 3. TensorFlow Keras Neural Network (Functional API)
#
# 架構: 兩層隱藏層 (64, 32)，與 sklearn MLP 保持一致

# %%
print("\n" + "=" * 50)
print("TensorFlow Keras Neural Network 訓練 (Functional API)")
print("=" * 50)

# 參考 Week 12 Functional API: Defining input → Connecting layers → Creating the model
# Step 1: Defining input
visible = Input(shape=(X_train.shape[1],))

# Step 2: Connecting layers - 使用 (64, 32) 架構
hidden1 = Dense(64, activation="tanh")(visible)
hidden2 = Dense(32, activation="tanh")(hidden1)
output = Dense(1, activation="sigmoid")(hidden2)

# Step 3: Creating the model
model = Model(inputs=visible, outputs=output)

model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"])

print("\n模型架構:")
model.summary()

# 訓練模型
start_tf = time.time()
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1
)
train_tf = time.time() - start_tf

# 預測
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

# %% [markdown]
# ### 模型比較表格

# %%
print("\n" + "=" * 50)
print("三種模型效能比較")
print("=" * 50)

# 建立比較表格
comparison = pd.DataFrame(
    {
        "模型": [
            "SVM (GridSearchCV)",
            "sklearn MLP (64,32)",
            "TensorFlow Keras (64,32)",
        ],
        "訓練時間 (秒)": [f"{train_svm:.3f}", f"{train_mlp:.3f}", f"{train_tf:.3f}"],
        "Accuracy": [f"{acc_svm:.4f}", f"{acc_mlp:.4f}", f"{acc_tf:.4f}"],
        "Precision": [f"{prec_svm:.4f}", f"{prec_mlp:.4f}", f"{prec_tf:.4f}"],
        "Recall": [f"{rec_svm:.4f}", f"{rec_mlp:.4f}", f"{rec_tf:.4f}"],
        "F1-Score": [f"{f1_svm:.4f}", f"{f1_mlp:.4f}", f"{f1_tf:.4f}"],
        "AUC": [f"{auc_svm:.4f}", f"{auc_mlp:.4f}", f"{auc_tf:.4f}"],
    }
)

print("\n")
print(comparison.to_string(index=False))

# 找出各指標最佳模型
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

# %% [markdown]
# ### ROC 曲線比較

# %%
# 計算三個模型的 ROC 曲線
# 注意：SVM 的 ROC 需要用原始的 0/1 標籤來計算
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
fpr_tf, tpr_tf, _ = roc_curve(y_test, y_proba_tf)

# 繪製 ROC 曲線
plt.figure(figsize=(10, 8))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.4f})", linewidth=2)
plt.plot(fpr_mlp, tpr_mlp, label=f"sklearn MLP (AUC = {auc_mlp:.4f})", linewidth=2)
plt.plot(fpr_tf, tpr_tf, label=f"TensorFlow Keras (AUC = {auc_tf:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="隨機猜測", linewidth=1)

plt.xlabel("偽陽性率 (False Positive Rate)", fontsize=12)
plt.ylabel("真陽性率 (True Positive Rate)", fontsize=12)
plt.title("三種模型 ROC 曲線比較", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### TensorFlow 訓練歷史視覺化

# %%
# 繪製訓練過程
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 準確率
axes[0].plot(history.history["accuracy"], label="訓練準確率", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="驗證準確率", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Accuracy", fontsize=11)
axes[0].set_title("TensorFlow Keras 訓練準確率", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 損失
axes[1].plot(history.history["loss"], label="訓練損失", linewidth=2)
axes[1].plot(history.history["val_loss"], label="驗證損失", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Loss", fontsize=11)
axes[1].set_title("TensorFlow Keras 訓練損失", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 結論
#
# #### 神經網路架構選擇
# - 本次實驗針對 22 個特徵的銀行定存資料，選用 **(64, 32)** 兩層隱藏層架構
# - 第一層 64 neurons 學習特徵組合，第二層 32 neurons 進行抽象表示
# - sklearn MLP 與 TensorFlow Keras 使用相同架構，便於公平比較
#
# #### 訓練時間比較
# - **SVM GridSearchCV**: 因需搜尋多組超參數，訓練時間最長
# - **sklearn MLP**: 訓練速度最快，適合快速原型開發
# - **TensorFlow Keras (Functional API)**: 訓練時間適中，架構彈性更高
#
# #### 預測效能比較
# - 比較 **Accuracy、Precision、Recall、F1-Score、AUC** 五項指標
# - SVM 使用 -1/1 標籤進行訓練，符合傳統 SVM 演算法的要求
# - Neural Network 模型使用 0/1 標籤，在已標準化的資料上表現良好
# - 兩層隱藏層架構對於中等規模特徵數（22 個）來說是合適的選擇
#
# #### 模型選擇建議
# 1. **追求穩定預測效能**: 使用 SVM GridSearchCV（雖然訓練較慢）
# 2. **需要快速訓練與驗證**: 使用 sklearn MLP
# 3. **需要彈性調整模型架構**: 使用 TensorFlow Keras Functional API
#
# #### Functional API 優勢
# - 可以處理多輸入、多輸出模型
# - 可以共享層 (layer sharing)
# - 更適合複雜的網路架構設計
#
# #### 實務考量
# - 若資料量大或特徵多，Neural Network 優勢更明顯
# - 若重視 Recall（不想漏掉有潛力的客戶），應選擇該指標較高的模型
# - 若重視 Precision（避免浪費行銷資源），應選擇該指標較高的模型
