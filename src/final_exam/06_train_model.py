import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

def main():
    print("=" * 70)
    print("基於多模態特徵的商品自動分類系統 - 模型訓練")
    print("=" * 70)

    # ==================== 1. 讀取處理好的特徵 ====================
    print("\n步驟 1: 讀取處理好的特徵")
    print("-" * 70)
    
    input_file = "output/processed_features.pkl"
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到檔案 {input_file}")
        print("請先執行 05_prepare_features.py")
        return

    with open(input_file, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    le = data["label_encoder"]
    
    print(f"特徵維度: {X.shape}")
    print(f"樣本數: {X.shape[0]}")
    print(f"類別數: {len(le.classes_)}")

    # ==================== 2. 分割訓練/測試集 ====================
    print("\n" + "=" * 70)
    print("步驟 2: 分割訓練/測試集")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"訓練集: {len(y_train)} 筆 ({len(y_train)/len(y)*100:.1f}%)")
    print(f"測試集: {len(y_test)} 筆 ({len(y_test)/len(y)*100:.1f}%)")

    # ==================== 3. 訓練多個模型 ====================
    print("\n" + "=" * 70)
    print("步驟 3: 訓練模型")
    print("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_model_info = None
    best_accuracy = 0
    best_y_pred = None

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"訓練 {name}...")
        print(f"{'='*70}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"準確率: {accuracy:.2%}")
        print("\n分類報告:")
        print(
            classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_info = (name, model)
            best_y_pred = y_pred

    # ==================== 4. 儲存最佳模型 ====================
    print(f"\n" + "=" * 70)
    print(f"最佳模型: {best_model_info[0]}")
    print(f"準確率: {best_accuracy:.2%}")
    print("=" * 70)

    # 更新儲存的資料，加入模型
    data["model"] = best_model_info[1]
    data["model_name"] = best_model_info[0]
    data["accuracy"] = best_accuracy
    
    # 移除原始大數據以節省空間 (如果需要的話，這裡選擇保留完整 pipeline 資訊)
    # del data["X"]
    # del data["y"]

    with open("output/best_model.pkl", "wb") as f:
        pickle.dump(data, f)
    print("模型已儲存到 output/best_model.pkl")

    # ==================== 5. 混淆矩陣視覺化 ====================
    print("\n生成混淆矩陣...")
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(
        f"混淆矩陣 - {best_model_info[0]}\n準確率: {best_accuracy:.2%}", fontsize=16, pad=20
    )
    plt.xlabel("預測分類", fontsize=12)
    plt.ylabel("實際分類", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("混淆矩陣已儲存到 output/confusion_matrix.png")

    # ==================== 6. 模型比較圖 ====================
    print("生成模型比較圖...")
    plt.figure(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = plt.bar(results.keys(), results.values(), color=colors[: len(results)])
    plt.ylabel("準確率", fontsize=12)
    plt.title("不同模型準確率比較", fontsize=16)
    plt.ylim(0, 1)
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/model_comparison.png", dpi=300, bbox_inches="tight")
    print("模型比較圖已儲存到 output/model_comparison.png")

    print("\n" + "=" * 70)
    print("訓練完成！")
    print("=" * 70)
    print(f"\n生成的檔案:")
    print(f"  - output/best_model.pkl (訓練好的模型)")
    print(f"  - output/confusion_matrix.png (混淆矩陣)")
    print(f"  - output/model_comparison.png (模型比較)")

if __name__ == "__main__":
    main()
