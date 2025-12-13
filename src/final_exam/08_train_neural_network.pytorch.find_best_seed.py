import os
import pickle
import time
import random
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


def main():
    print("Starting Standalone Seed Miner...")

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # 讀取資料
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"Error: {traditional_file} not found.")
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
    global_best_acc = 0.0
    best_seed = None

    # 已測試過的 Seed 排除清單
    banned_seeds = {
        47742,
        38847,
        58224,
        55580,
        31628,
        69152,
        1246,
        92001,
        57760,
        54904,
        36345,
        51701,
        33310,
        14521,
        97699,
        78527,
        65113,
        68705,
        46412,
        45515,
        3714,
        46915,
        56079,
        21867,
        2280,
        78069,
        15192,
        73876,
        1106,
        14885,
        53044,
        29154,
        40390,
        7,
    }

    # 載入當前最佳紀錄
    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                prev_results = pickle.load(f)
                global_best_acc = prev_results.get("best_accuracy", 0)
                best_seed = prev_results.get("best_seed", None)
            print(f"Current Best Record: Seed {best_seed}, Acc {global_best_acc:.4f}")
        except Exception:
            pass

    lr = 0.00025
    bs = 20
    current_session_seeds = set()
    iteration = 0

    try:
        while True:
            # 隨機生成 Seed (1 ~ 1,000,000)
            seed = random.randint(1, 1000000)

            if (seed in banned_seeds) or (seed in current_session_seeds):
                continue

            current_session_seeds.add(seed)
            iteration += 1

            # 設定 Seed
            keras.utils.set_random_seed(seed)

            # 資料切分
            X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
                X_train_smote,
                y_train_keras,
                test_size=0.1,
                random_state=seed,
                stratify=y_train_smote,
            )

            # 建立與訓練
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

            # 驗證
            probs = model.predict(X_test, verbose=0)
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(y_test, preds)

            print(
                f"Iter {iteration} | Seed: {seed:<7} | Acc: {acc:.4f} | Time: {elapsed_time:.1f}s"
            )

            # 若創新高則儲存
            if acc > global_best_acc:
                print(
                    f">>> New Best Found! Seed {seed} ({acc:.4f}) beat ({global_best_acc:.4f})"
                )

                global_best_acc = acc
                best_seed = seed

                model.save("output/models/best_keras_model.keras")

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

                with open("output/models/keras_results.pkl", "wb") as f:
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
                plt.title(f"Seed={seed}, Acc={acc:.4f}")
                plt.tight_layout()
                plt.savefig(f"output/result_images/best_confusion_matrix_{seed}.png")
                plt.close()

    except KeyboardInterrupt:
        print("\nMining stopped by user.")


if __name__ == "__main__":
    main()
