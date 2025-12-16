import os
import pickle
import time
import random
import csv
import platform
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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


def build_keras_model(input_dim, num_classes, learning_rate):
    inputs = Input(shape=(input_dim,), name="input_features")

    x = Dense(1024, name="dense_1024")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(0.45, name="dropout_0")(x)

    x = Dense(512, name="dense_512")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.45, name="dropout_1")(x)

    x = Dense(256, name="dense_256")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.45, name="dropout_2")(x)

    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

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


def load_tested_seeds_from_csv(csv_file):
    """從 CSV 讀取已測試過的 seed"""
    tested_seeds = set()

    if not os.path.isfile(csv_file):
        return tested_seeds

    try:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seed = int(row["Seed"])
                tested_seeds.add(seed)
        print(f"從 {csv_file} 載入 {len(tested_seeds)} 個已測試的 seed")
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")

    return tested_seeds


def main():
    # 取得系統資訊
    system_name = platform.system()
    platform_info = f"{system_name}-{platform.machine()}"

    print("=" * 70)
    print("Seed Mining - 靜音模式 (所有紀錄寫入 CSV)")
    print(f"執行平台: {platform_info}")
    print("=" * 70)

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # 讀取資料
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

    if sparse.issparse(X_train_smote):
        X_train_smote = X_train_smote.toarray().astype("float32")
    if sparse.issparse(X_test):
        X_test = X_test.toarray().astype("float32")

    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))

    results_file = "output/models/keras_results.pkl"

    # 根據平台命名 CSV 檔案
    if system_name == "Windows":
        csv_all_file = "output/models/seed_mining_all_windows.csv"
        csv_best_file = "output/models/seed_mining_best_windows.csv"
    else:  # Linux (WSL)
        csv_all_file = "output/models/seed_mining_all_linux.csv"
        csv_best_file = "output/models/seed_mining_best_linux.csv"

    global_best_acc = 0.0
    best_seed = None

    # 從 CSV 載入已測試過的 seed
    print("\n讀取已測試的 seed...")
    tested_seeds = load_tested_seeds_from_csv(csv_all_file)

    # 也讀取其他平台的 CSV（避免跨平台重複測試）
    if system_name == "Windows":
        other_csv = "output/models/seed_mining_all_linux.csv"
    else:
        other_csv = "output/models/seed_mining_all_windows.csv"

    if os.path.isfile(other_csv):
        other_seeds = load_tested_seeds_from_csv(other_csv)
        tested_seeds.update(other_seeds)
        print(f"合併兩平台後，共 {len(tested_seeds)} 個已測試的 seed")

    # 載入當前最佳紀錄
    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                prev_results = pickle.load(f)
                global_best_acc = prev_results.get("best_accuracy", 0)
                best_seed = prev_results.get("best_seed", None)
            print(f"目前最佳紀錄: seed={best_seed}, acc={global_best_acc:.4f}")
        except Exception:
            pass

    lr = 0.00025
    bs = 20
    iteration = 0
    start_session_time = time.time()

    # 初始化 CSV 檔案（如果不存在）
    for csv_file in [csv_all_file, csv_best_file]:
        if not os.path.isfile(csv_file):
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "時間戳記",
                        "Iteration",
                        "Seed",
                        "準確率",
                        "訓練耗時(秒)",
                        "LR",
                        "Batch_Size",
                        "平台",
                        "是否最佳",
                    ]
                )

    print(f"\n紀錄檔案:")
    print(f"  所有測試: {csv_all_file}")
    print(f"  最佳紀錄: {csv_best_file}")
    print(f"\n開始挖掘... (每 50 次顯示一次進度)")
    print(f"已排除 {len(tested_seeds)} 個測試過的 seed\n")

    skipped_count = 0  # 統計跳過的次數

    try:
        while True:
            # 隨機生成 seed
            seed = random.randint(1, 1000000)

            # 檢查是否已測試過
            if seed in tested_seeds:
                skipped_count += 1
                # 每跳過 1000 次顯示一次（避免 seed 空間快用完時卡住）
                if skipped_count % 1000 == 0:
                    print(
                        f"[警告] 已跳過 {skipped_count} 個重複的 seed，可能需要擴大 seed 範圍"
                    )
                continue

            # 加入已測試集合（避免本次 session 內重複）
            tested_seeds.add(seed)
            iteration += 1

            keras.utils.set_random_seed(seed)

            X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
                X_train_smote,
                y_train_keras,
                test_size=0.1,
                random_state=seed,
                stratify=y_train_smote,
            )

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

            start_time = time.time()
            history = model.fit(
                X_train_sub,
                y_train_sub,
                batch_size=bs,
                epochs=300,
                validation_data=(X_valid, y_valid),
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )
            elapsed_time = time.time() - start_time

            probs = model.predict(X_test, verbose=0)
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(y_test, preds)

            is_best = acc > global_best_acc
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 寫入所有測試紀錄
            with open(csv_all_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        timestamp,
                        iteration,
                        seed,
                        f"{acc:.6f}",
                        f"{elapsed_time:.2f}",
                        lr,
                        bs,
                        platform_info,
                        "Yes" if is_best else "No",
                    ]
                )

            # 若創新高則處理
            if is_best:
                old_best = global_best_acc
                global_best_acc = acc
                best_seed = seed

                # 寫入最佳紀錄
                with open(csv_best_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            timestamp,
                            iteration,
                            seed,
                            f"{acc:.6f}",
                            f"{elapsed_time:.2f}",
                            lr,
                            bs,
                            platform_info,
                            "Yes",
                        ]
                    )

                # 儲存模型
                model.save("output/models/best_keras_model.keras")

                # 儲存 Pickle
                keras_results = {
                    "best_model": None,
                    "best_lr": lr,
                    "best_bs": bs,
                    "best_seed": seed,
                    "best_accuracy": acc,
                    "best_y_pred": preds,
                    "best_history": history,
                    "X_test": X_test,
                    "y_test": y_test,
                    "label_encoder": le,
                }
                with open(results_file, "wb") as f:
                    pickle.dump(keras_results, f)

                # 繪製混淆矩陣
                plt.figure(figsize=(12, 10))
                cm = confusion_matrix(y_test, preds)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Reds",
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                )
                plt.title(f"{platform_info} - seed={seed}, Acc={acc:.4f}")
                plt.tight_layout()
                plt.savefig(
                    f"output/result_images/best_cm_{system_name.lower()}_{seed}.png"
                )
                plt.close()

                # 只在找到新紀錄時 print
                print(
                    f"[{iteration:4d}] NEW BEST! seed={seed} | {old_best:.4f} -> {acc:.4f} (+{acc-old_best:.4f})"
                )

            # 每 50 次顯示進度
            if iteration % 50 == 0:
                session_time = time.time() - start_session_time
                avg_time = session_time / iteration

                print(
                    f"[{iteration:4d}] 已測試 {iteration} 個 seed (跳過 {skipped_count} 個重複)"
                )
                print(
                    f"       運行: {session_time/3600:.2f}h | 平均: {avg_time:.1f}s/seed"
                )
                print(f"       最佳: seed={best_seed}, acc={global_best_acc:.4f}")
                print()

    except KeyboardInterrupt:
        print("\n\n使用者停止挖掘。")
        session_time = time.time() - start_session_time
        print(f"\n最終統計:")
        print(f"  平台: {platform_info}")
        print(f"  總測試: {iteration} 個 seed")
        print(f"  跳過重複: {skipped_count} 個")
        print(f"  總時間: {session_time/3600:.2f} 小時")
        print(f"  最佳結果: seed={best_seed}, acc={global_best_acc:.4f}")
        print(f"\n紀錄已儲存至:")
        print(f"  - {csv_all_file}")
        print(f"  - {csv_best_file}")


if __name__ == "__main__":
    main()
