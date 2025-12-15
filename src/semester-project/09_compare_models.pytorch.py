import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import keras

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 70)
    print("模型比較分析 (整合 Seed Mining 結果)")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取所有模型結果 ====================
    print("\n步驟 1: 讀取所有模型結果")
    print("-" * 70)

    # 1.1 讀取傳統模型
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    # 1.2 讀取 Keras 模型結果 (Pickle)
    keras_file = "output/models/keras_results.pkl"
    if not os.path.exists(keras_file):
        print(f"錯誤: 找不到 {keras_file}")
        return

    with open(keras_file, "rb") as f:
        keras_data = pickle.load(f)

    print("成功讀取所有模型結果")

    # ==================== 2. 整合結果 ====================
    print("\n步驟 2: 整合模型結果")
    print("-" * 70)

    all_results = {}
    all_times = {}

    # 2.1 加入傳統模型數據
    all_results.update(trad_data["results"])
    all_times.update(trad_data["training_times"])

    # 2.2 處理 Keras 數據 (適應 Seed Mining 的結構)
    # [關鍵修正 2] 這裡的 key 應該是 "best_seed"，不是 "best_state"
    best_lr = keras_data.get("best_lr")
    best_bs = keras_data.get("best_bs")
    best_seed = keras_data.get("best_seed", "Unknown")
    best_acc = keras_data["best_accuracy"]

    # 設定顯示名稱
    keras_name = f"Neural Net (Seed={best_seed})"
    all_results[keras_name] = best_acc

    # 2.3 獲取 Keras 訓練時間
    keras_time = 0

    if "train_times" in keras_data and isinstance(keras_data["train_times"], dict):
        # 舊版 Grid Search 結構
        key = (best_lr, best_bs)
        keras_time = keras_data["train_times"].get(key, 0)
    else:
        # 新版 Seed Miner 結構 - 嘗試從 CSV 讀取
        # [關鍵修正 3] 檔名修正為 seed_mining_history.csv
        csv_path = "output/models/seed_mining_history.csv"

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # [關鍵修正 4] 欄位名稱修正為 "Seed"
                # 確保 Seed 欄位格式一致 (轉字串比對)
                seed_row = df[df["Seed"].astype(str) == str(best_seed)]

                if not seed_row.empty:
                    # 取最後一筆 (如果同一個 Seed 跑多次)
                    keras_time = float(seed_row.iloc[-1]["訓練耗時(秒)"])
                    print(
                        f"從 CSV 找到 Seed {best_seed} 的訓練時間: {keras_time:.2f} 秒"
                    )
                else:
                    print(f"CSV 中找不到對應 Seed ({best_seed}) 的時間，設為 0")
            except Exception as e:
                print(f"讀取 CSV 時間失敗: {e}")
        else:
            print(
                "找不到 CSV 歷史紀錄 (output/models/seed_mining_history.csv)，訓練時間設為 0"
            )

    all_times[keras_name] = keras_time

    # ==================== 3. 找出最佳模型 ====================
    best_model_name = max(all_results, key=all_results.get)
    best_accuracy = all_results[best_model_name]

    print(f"最佳模型: {best_model_name}")
    print(f"準確率: {best_accuracy:.2%}")

    # ==================== 4. 生成比較圖表 ====================
    print("\n步驟 3: 生成比較圖表")
    print("-" * 70)

    # 4.1 準確率比較圖
    plt.figure(figsize=(14, 6))
    # 動態生成顏色 (Keras 用紅色，其他用藍/綠)
    colors = [
        "#e74c3c" if "Neural" in name else "#3498db" for name in all_results.keys()
    ]

    bars = plt.bar(all_results.keys(), all_results.values(), color=colors, alpha=0.8)
    plt.ylabel("準確率", fontsize=12)
    plt.title("不同模型準確率比較", fontsize=16)
    plt.ylim(0.7, 0.9)  # 設定 Y 軸範圍讓差異更明顯

    for i, (name, acc) in enumerate(all_results.items()):
        plt.text(
            i, acc + 0.002, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold"
        )

    plt.xticks(rotation=20, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("output/result_images/model_comparison.png", dpi=300)
    print("準確率比較圖已儲存")
    plt.close()

    # 4.2 訓練時間比較圖
    fig, ax = plt.subplots(figsize=(14, 6))
    model_names = list(all_times.keys())
    times = list(all_times.values())

    bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.8)

    ax.set_ylabel("訓練時間 (秒)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_title("模型訓練時間比較", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for i, t in enumerate(times):
        time_text = f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m"
        ax.text(i, t + max(times) * 0.02, time_text, ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("output/result_images/training_time_comparison.png", dpi=300)
    print("訓練時間比較圖已儲存")
    plt.close()

    # 4.3 效能/時間權衡圖
    plt.figure(figsize=(10, 6))
    accuracies = [all_results[name] for name in model_names]

    plt.scatter(
        times, accuracies, s=300, c=colors, alpha=0.6, edgecolors="black", linewidth=2
    )

    for i, name in enumerate(model_names):
        short_name = name.split("(")[0].strip()
        plt.annotate(
            short_name,
            (times[i], accuracies[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        )

    plt.xlabel("訓練時間 (秒)")
    plt.ylabel("測試準確率")
    plt.title("模型效能 vs 訓練時間權衡")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/result_images/performance_time_tradeoff.png", dpi=300)
    print("效能/時間權衡圖已儲存")
    plt.close()

    # ==================== 5. 生成摘要報告 ====================
    print("\n步驟 4: 生成摘要報告")

    summary = []
    summary.append("=" * 70)
    summary.append("模型比較摘要報告")
    summary.append("=" * 70)
    summary.append("")

    summary.append("1. 準確率排名:")
    for i, (name, acc) in enumerate(
        sorted(all_results.items(), key=lambda x: x[1], reverse=True), 1
    ):
        summary.append(f"   {i}. {name}: {acc:.2%}")

    summary.append("")
    summary.append("2. 最佳模型詳情:")
    summary.append(f"   模型: {best_model_name}")
    summary.append(f"   準確率: {best_accuracy:.2%}")
    summary.append(f"   訓練時間: {all_times.get(best_model_name, 0):.2f} 秒")

    # 輸出並存檔
    for line in summary:
        print(line)

    with open("output/model_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    # ==================== 6. 儲存最佳模型 (整合包) ====================
    print("\n步驟 5: 儲存最佳模型整合包")
    print("-" * 70)

    best_model_data = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "training_time": all_times.get(best_model_name, 0),
    }

    if "Neural" in best_model_name:
        # 特別處理 Keras 模型
        print("最佳模型為 Keras，正在載入實體模型檔案...")
        try:
            # 載入真正的模型物件
            keras_model_path = "output/models/best_keras_model.keras"
            loaded_model = keras.models.load_model(keras_model_path)

            best_model_data["model"] = None  # 避免 pickle 失敗
            best_model_data["model_path"] = keras_model_path  # 存路徑
            best_model_data["y_pred"] = keras_data["best_y_pred"]
            best_model_data["label_encoder"] = keras_data["label_encoder"]
            print("Keras 模型資訊已整合 (模型本體建議讀取 .keras 檔案)")
        except Exception as e:
            print(f"載入 Keras 模型失敗: {e}")
    else:
        # 傳統模型
        model_type = best_model_name
        best_model_data["model"] = trad_data["models"][model_type]
        best_model_data["y_pred"] = trad_data["predictions"][model_type]
        best_model_data["label_encoder"] = trad_data["label_encoder"]

    best_model_file = "output/models/final_best_model_info.pkl"
    with open(best_model_file, "wb") as f:
        pickle.dump(best_model_data, f)

    print(f"\n最佳模型整合資訊已儲存至: {best_model_file}")


if __name__ == "__main__":
    main()
