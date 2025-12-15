# ==========================================
# 這隻程式碼嘗試使用 Keras 3 語法取代 tensorflow.keras
# 原因: 因為 TensorFlow 在 RTX50 系列顯示卡上的支援較差
# pytorch 對新顯示卡支援較好，因此改用 Keras 3 + PyTorch Backend
# 主要差異在於 import 的方式不同
# 其餘程式碼介面基本上跟 tensorflow.keras 相同
# ==========================================
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False

print(f"目前使用的 Keras 後端: {keras.backend.backend()}")


def build_keras_model(input_dim, num_classes, learning_rate):
    """
    建立 Keras 神經網路模型 (PyTorch Backend)。
    架構：512 -> 256 -> 128 -> 64 -> Softmax
    """
    inputs = Input(shape=(input_dim,), name="input_features")

    # 第一層: 1024 neurons
    x = Dense(1024, name="dense_1024")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(0.45, name="dropout_0")(x)

    # 第一層: 512 neurons
    x = Dense(512, name="dense_512")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.45, name="dropout_1")(x)

    # 第二層: 256 neurons
    x = Dense(256, name="dense_256")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.45, name="dropout_2")(x)

    # 第三層: 128 neurons
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

    # 第四層: 64 neurons
    x = Dense(64, name="dense_64")(x)
    x = BatchNormalization(name="batchnorm_4")(x)
    x = Activation("relu", name="activation_4")(x)
    x = Dropout(0.3, name="dropout_4")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="product_classifier_keras")
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 70)
    print("神經網路模型訓練 (Keras 3 + PyTorch Backend)")
    print("=" * 70)

    TARGET_SEED = 232268
    print(f">>> 最佳 random_seed: {TARGET_SEED}")
    keras.utils.set_random_seed(TARGET_SEED)

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取資料 ====================
    print("\n步驟 1: 讀取資料")
    traditional_file = "output/models/traditional_models.pkl"

    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    X_train_smote = trad_data["X_train_smote"]
    y_train_smote = trad_data["y_train_smote"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 轉為 Dense Array
    if sparse.issparse(X_train_smote):
        X_train_smote = X_train_smote.toarray().astype("float32")
    if sparse.issparse(X_test):
        X_test = X_test.toarray().astype("float32")

    print(f"訓練集: {X_train_smote.shape}")
    print(f"測試集: {X_test.shape}")

    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))

    # ==================== 2. 訓練模型 ====================
    print("\n步驟 2: 開始訓練 (使用最佳參數)")

    # 固定參數
    lr = 0.00025
    bs = 20

    keras_histories = {}
    keras_accuracies = {}
    keras_train_times = {}

    best_acc = 0
    best_params = {}
    best_model = None
    best_y_pred = None

    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train_smote,
        y_train_keras,
        test_size=0.1,
        random_state=TARGET_SEED,
        stratify=y_train_smote,
    )

    print(f"\n--- Training: Seed={TARGET_SEED}, lr={lr}, bs={bs} ---")

    model = build_keras_model(X_train_smote.shape[1], len(le.classes_), lr)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        min_delta=0.0005,
    )

    start = time.time()

    history = model.fit(
        X_train_sub,
        y_train_sub,
        batch_size=bs,
        epochs=300,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    train_time = time.time() - start

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)

    print(f"結果 -> 準確率: {acc:.2%}, 耗時: {train_time:.1f}s")

    # 記錄
    key = (lr, bs)
    keras_accuracies[key] = acc
    keras_histories[key] = history
    keras_train_times[key] = train_time

    best_acc = acc
    best_params = {"lr": lr, "bs": bs}
    best_model = model
    best_y_pred = preds

    # ==================== 3. 儲存結果 ====================
    print("\n步驟 3: 儲存訓練結果")

    # 1. 儲存 .keras 模型 (最重要)
    best_model.save("output/models/best_keras_model.keras")

    # 2. 儲存 .pkl 結果包
    keras_results = {
        "best_model": None,  # 避免 pickle 錯誤，不存物件
        "best_lr": best_params["lr"],
        "best_bs": best_params["bs"],
        "best_seed": TARGET_SEED,  # [新增] 記錄 Seed
        "best_accuracy": best_acc,
        "best_y_pred": best_y_pred,
        "histories": keras_histories,
        "accuracies": keras_accuracies,
        "train_times": keras_train_times,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
    }

    with open("output/models/keras_results.pkl", "wb") as f:
        pickle.dump(keras_results, f)

    print("模型與結果已儲存。")

    # ==================== 4. 視覺化 ====================
    print("\n步驟 4: 生成圖表")

    # 4.1 Training History
    best_hist = keras_histories[(best_params["lr"], best_params["bs"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(best_hist.history["loss"], label="Train Loss")
    axes[0].plot(best_hist.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Loss Curve (Seed={TARGET_SEED})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(best_hist.history["accuracy"], label="Train Acc")
    axes[1].plot(best_hist.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/result_images/keras_training_history.png", dpi=300)
    plt.close()

    # 4.2 Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"Confusion Matrix (Seed={TARGET_SEED})\nAccuracy: {best_acc:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/result_images/keras_confusion_matrix.png", dpi=300)
    plt.close()

    print("圖表已生成完成。")


if __name__ == "__main__":
    main()
