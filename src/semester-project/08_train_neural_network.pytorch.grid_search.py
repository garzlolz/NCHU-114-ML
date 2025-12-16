import os
import pickle
import time
import csv
import platform
from datetime import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # 新增

# 指定 PyTorch 為後端
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from utils.cross_platform_config import set_matplotlib_font

# 字型設定
font_name = set_matplotlib_font()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes, learning_rate, dropout_list):
    """
    簡化版神經網路模型
    架構：256 → 128 → 64 → Softmax
    dropout_list: [d0, d1, d2] (只有 3 個)
    """
    d0, d1, d2 = dropout_list

    inputs = Input(shape=(input_dim,), name="input_features")

    # 第一層：256 neurons
    x = Dense(256, name="dense_256")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(d0, name="dropout_0")(x)

    # 第二層：128 neurons
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(d1, name="dropout_1")(x)

    # 第三層：64 neurons
    x = Dense(64, name="dense_64")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(d2, name="dropout_2")(x)

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
    system_name = platform.system()
    platform_info = f"{system_name}-{platform.machine()}"

    print("=" * 70)
    print("Keras Neural Network - Grid Search Hyperparameters")
    print(f"執行平台: {platform_info}")
    print("模型結構: 256 → 128 → 64")
    print("=" * 70)

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # 讀取前一階段的資料
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        print("請先執行 07_train_traditional_model.py")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    # 載入原始訓練集（未 SMOTE）
    X_train = trad_data["X_train"]
    y_train = trad_data["y_train"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 轉換格式
    if sparse.issparse(X_train):
        X_train = X_train.toarray().astype("float32")
    if sparse.issparse(X_test):
        X_test = X_test.toarray().astype("float32")

    print("原始訓練集尺寸:", X_train.shape)
    print("測試集尺寸:", X_test.shape)

    num_classes = len(le.classes_)

    results_file = "output/models/keras_results.pkl"
    csv_all_file = "output/models/keras_grid_search_all.csv"

    # 固定一個隨機種子
    BASE_SEED = 232268
    keras.utils.set_random_seed(BASE_SEED)

    # ========== 先分割驗證集（在 SMOTE 之前） ==========
    print("\n分割訓練/驗證集（在 SMOTE 之前）...")
    X_train_orig, X_valid, y_train_orig, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,  # 20% 驗證集
        random_state=BASE_SEED,
        stratify=y_train,
    )

    print(f"訓練子集（未 SMOTE）: {X_train_orig.shape}")
    print(f"驗證集: {X_valid.shape}")

    # 轉換驗證集標籤為 one-hot
    y_valid_keras = to_categorical(y_valid, num_classes=num_classes)

    # 定義要搜尋的超參數網格
    lr_list = [1e-4, 2e-4, 2.5e-4, 3e-4, 5e-4]  # 5 個
    bs_list = [16, 20, 32, 48, 64]  # 5 個
    dropout_grid = [
        [0.3, 0.3, 0.2],  # 與 train 預設一致
        [0.4, 0.4, 0.3],
        [0.35, 0.35, 0.25],
        [0.25, 0.25, 0.15],
        [0.5, 0.5, 0.4],
    ]  # 5 組
    # 總組合 = 5 × 5 × 5 = 125 組

    param_combinations = list(
        itertools.product(lr_list, bs_list, range(len(dropout_grid)))
    )
    total_combos = len(param_combinations)

    print("\nGrid Search 設定：")
    print(f"  learning_rate 選項: {lr_list}")
    print(f"  batch_size 選項  : {bs_list}")
    print(f"  dropout 組數    : {len(dropout_grid)} 組")
    print(f"  dropout 配置    : {dropout_grid}")
    print(f"  總組合數        : {total_combos}")
    print("=" * 70)

    # 準備 CSV
    if not os.path.isfile(csv_all_file):
        with open(csv_all_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "時間戳記",
                    "Index",
                    "Learning_Rate",
                    "Batch_Size",
                    "Dropout_Config_ID",
                    "Accuracy",
                    "訓練耗時(秒)",
                    "平台",
                ]
            )
        print(f"已建立紀錄檔: {csv_all_file}")

    global_best_acc = 0.0
    best_params = None
    best_preds = None
    best_history = None

    start_session_time = time.time()

    for idx, (lr, bs, d_id) in enumerate(param_combinations, start=1):
        dropouts = dropout_grid[d_id]
        print(
            f"\n[{idx:03d}/{total_combos}] 開始訓練 - "
            f"lr={lr}, bs={bs}, dropout={dropouts}"
        )

        # ========== 對訓練子集做 SMOTE ==========
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_orig, y_train_orig)

        # 轉換為 one-hot
        y_train_keras = to_categorical(y_train_smote, num_classes=num_classes)

        model = build_keras_model(
            input_dim=X_train_smote.shape[1],
            num_classes=num_classes,
            learning_rate=lr,
            dropout_list=dropouts,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,  # 簡化結構用較小的 patience
            restore_best_weights=True,
            min_delta=0.001,
        )

        start_time = time.time()
        history = model.fit(
            X_train_smote,
            y_train_keras,
            batch_size=bs,
            epochs=100,  # 簡化結構用較少 epochs
            validation_data=(X_valid, y_valid_keras),
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        )
        elapsed_time = time.time() - start_time

        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)

        print(f" -> 測試集準確率: {acc:.4f}")
        print(f" -> 訓練耗時    : {elapsed_time:.2f} 秒")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_all_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    idx,
                    lr,
                    bs,
                    d_id,
                    f"{acc:.6f}",
                    f"{elapsed_time:.2f}",
                    platform_info,
                ]
            )

        if acc > global_best_acc:
            old_best = global_best_acc
            global_best_acc = acc
            best_params = {
                "learning_rate": lr,
                "batch_size": bs,
                "dropout_config_id": d_id,
                "dropouts": dropouts,
            }
            best_preds = preds
            best_history = history

            print(
                f" *** NEW BEST *** acc: {old_best:.4f} -> {global_best_acc:.4f} "
                f"(lr={lr}, bs={bs}, dropout={dropouts})"
            )

            # 儲存最佳模型
            model.save("output/models/best_keras_model.keras")

            # 儲存結果到 pickle
            keras_results = {
                "best_model": None,
                "best_lr": lr,
                "best_bs": bs,
                "best_dropout_id": d_id,
                "best_dropouts": dropouts,
                "best_accuracy": acc,
                "best_y_pred": best_preds,
                "best_history": best_history,
                "X_test": X_test,
                "y_test": y_test,
                "label_encoder": le,
            }
            with open(results_file, "wb") as f:
                pickle.dump(keras_results, f)

            # 繪製混淆矩陣
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, best_preds)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Reds",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
            )
            plt.title(
                f"Keras Grid Search Best\n"
                f"Acc={global_best_acc:.4f}, lr={lr}, bs={bs}, dropout={dropouts}"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig("output/result_images/keras_best_confusion_matrix.png")
            plt.close()

    # 全部跑完後，畫訓練曲線
    if best_history is not None and best_params is not None:
        print("\n繪製最佳組合的訓練曲線...")

        hist = best_history
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist.history["loss"], label="Train Loss")
        plt.plot(hist.history["val_loss"], label="Val Loss")
        plt.title("Loss Curve (Best Hyperparameters)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(hist.history["accuracy"], label="Train Acc")
        plt.plot(hist.history["val_accuracy"], label="Val Acc")
        plt.title("Accuracy Curve (Best Hyperparameters)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("output/result_images/keras_best_training_history.png", dpi=300)
        plt.close()

        session_time = time.time() - start_session_time
        print("\nGrid Search 完成！")
        print(f"  總組合數: {total_combos}")
        print(f"  總耗時  : {session_time/3600:.2f} 小時")
        print(
            "  最佳組合: "
            f"lr={best_params['learning_rate']}, "
            f"bs={best_params['batch_size']}, "
            f"dropout={best_params['dropouts']}, "
            f"acc={global_best_acc:.4f}"
        )
        print(f"  紀錄檔案: {csv_all_file}")
        print(f"  模型 / 結果: best_keras_model.keras, keras_results.pkl")
    else:
        print("\nGrid Search 未找到任何有效結果，請檢查設定。")


if __name__ == "__main__":
    main()
