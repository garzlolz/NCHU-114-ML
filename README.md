## NCHU 114, 資管系在職碩專班 專題與功課

### 專案總覽

本專案旨在對電商商品進行多模態分類，整合**文字特徵 (TF-IDF)**、**圖片特徵 (HOG+顏色直方圖)** 和**價格特徵**，並使用神經網路 (Keras 3 + PyTorch) 和傳統機器學習模型 (Random Forest, Logistic Regression) 進行分類模型的訓練與比較。

#### 專題目錄結構

```text
. (專案根目錄)
├── .gitignore                                              # Git 忽略設定檔
├── environment.pytorch.yml                                 # PyTorch 環境設定檔 (Conda)
├── environment.tensorflow.yml                              # TensorFlow 環境設定檔 (備用)
└── src/                                                    # 原始碼資料夾
    ├── data/                                               # 資料存放區
    ├── semester-project/                                   # 學期專題主要程式碼
    │   ├── output/                                         # 輸出檔案 (模型、圖片、CSV)
    │   ├── utils/                                          # 通用工具函式庫
    │   ├── 01_crawl_savesafe.py                            # 爬蟲：抓取大買家商品資料
    │   ├── 02_download_images.py                           # 爬蟲：下載商品圖片
    │   ├── 03_clear_sold_out_product.py                    # 資料清理：移除售完商品
    │   ├── 04_extract_image_features.py                    # 特徵工程：提取圖片特徵 (HOG+色彩)
    │   ├── 05_tune_tfidf_params.py                         # 參數調優：TF-IDF 參數最佳化
    │   ├── 06_prepare_features.name_only.py                # 特徵工程：僅處理名稱特徵版本
    │   ├── 06_prepare_features.py                          # 特徵工程：整合文字、圖片與價格特徵
    │   ├── 07_train_traditional_model.py                   # 模型訓練：傳統機器學習 (RF, LR)
    │   ├── 08_train_neural_network.py                      # 模型訓練：神經網路 (TensorFlow/Keras)
    │   ├── 08_train_neural_network.pytorch.py              # 模型訓練：神經網路 (PyTorch Backend)
    │   ├── 08_train_neural_network.pytorch.state_miner.py  # 模型訓練：PyTorch 找 best random state
    │   ├── 09_compare_models.py                            # 模型評估：比較不同模型效能
    │   ├── 09_compare_models.pytorch.py                    # 模型評估：用 PyTorch 比較模型效能
    │   ├── 10_ensemble_prediction.pytorch.py               # 模型應用：集成學習預測
    │   ├── 11_predict_single_product.py                    # 模型應用：單一商品預測功能
    │   └── 12_baseline_lookup_table.py                     # 基線模型：查表法 Baseline
    └── week[1-12]/                                         # 各週次課程作業 (week1, week7...)
```

-----

## 一、環境目標與建議方案

本專案在 **Windows 11 + WSL2 + RTX 5070** 硬體環境下執行。由於 RTX 50 系列（Blackwell 架構）較新，舊版 TensorFlow 支援度不佳，本專案的 **主要方案** 採用 **Keras 3 + PyTorch 後端** 進行加速。

### 方案 A (推薦)：本機 Conda 環境 (PyTorch + Keras 3)

此方案能發揮 RTX 5070 的最大效能，且設定相對簡潔。

#### 1\. 前置準備：CUDA 支援

由於 GPU 驅動程式支援至 **CUDA 13.1**，為確保相容性與效能，建議安裝 **CUDA 13.0** 對應版本的 PyTorch。

#### 2\. 建立 Conda 虛擬環境

由於 `environment.pytorch.yml` 中的 PyTorch 相關項目已移除，故執行以下指令可建立包含所有非 PyTorch 依賴項的基礎環境。

```bash
# 1. 建立或更新環境
conda env update -f environment.pytorch.yml

# 2. 啟動環境
conda activate py312_keras_torch
```

> **注意：PyTorch Nightly Build 的特殊處理**
> 由於 `environment.pytorch.yml` 中鎖定了特定的開發者版本，該版本不在預設的 PyPI 網站上，會導致 Conda 安裝失敗。
>
> **【手動安裝步驟】**
>
> 1.  先移除 `environment.pytorch.yml` 中 `pip:` 區塊的 `torch`、`torchvision` 兩行，讓 Conda 完成基礎環境安裝。
> 2.  進入環境後，**手動執行**安裝指令，並指定 PyTorch 的 CUDA 13.0 專用下載網址：
>
> <!-- end list -->

> ```bash
> # 在 (py312_keras_torch) 環境中執行
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
> ```
#### 3\. 設定 Keras 後端

程式碼已在訓練腳本 (`08_train_neural_network.pytorch.py`) 的最上方設定環境變數，將 Keras 3 的運算核心切換為 PyTorch：

```python
# 位於 08_train_neural_network.pytorch.py
import os
os.environ["KERAS_BACKEND"] = "torch"
# ... (其他程式碼)
```

#### 4\. 驗證 GPU 與 Keras 後端

請在 `(py312_keras_torch)` 環境中執行以下指令確認：

| 驗證項目         | 指令                                                                           | 預期輸出               |
| :--------------- | :----------------------------------------------------------------------------- | :--------------------- |
| **PyTorch CUDA** | `python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"`   | `CUDA 可用: True`      |
| **Keras 後端**   | `python -c "import keras; print(f'Keras Backend: {keras.backend.backend()}')"` | `Keras Backend: torch` |

#### 5\. 執行訓練

```bash
cd src/semester-project
python 08_train_neural_network.pytorch.py
```

-----

## 二、方案 B (備用)：NVIDIA Docker 容器 (TensorFlow)

若必須使用 TensorFlow 進行訓練，建議使用 NVIDIA 官方的 TensorFlow Docker 容器，以解決 Pip 版 TensorFlow 與新 GPU 架構的相容性問題。

### 1\. 本機 Pip 版 TensorFlow 失敗原因紀錄

  * **問題：** 在 WSL Conda 環境中，使用 `pip` 安裝的 TensorFlow 搭配 CUDA 相關套件，執行訓練時會出現相容性錯誤，例如 `CUDA_ERROR_INVALID_PTX`。
  * **原因：** Pip 版本的 TensorFlow 缺少針對 Blackwell 架構 (Compute Capability 12.0) 編譯的 CUDA binary，與 RTX 50 系列的相容性不佳。

### 2\. VS Code Dev Container 設定

此方案利用 VS Code 的 Dev Containers 功能，自動在 Docker 容器內建立開發環境：

  * **基礎映像檔 (Image)：** `nvcr.io/nvidia/tensorflow:25.01-tf2-py3`
  * **GPU 設定：** `runArgs: ["--gpus=all"]`

#### `.devcontainer/devcontainer.json` 內容

```json
{
  "name": "RTX5070-TensorFlow",
  "image": "nvcr.io/nvidia/tensorflow:25.01-tf2-py3",
  "runArgs": ["--gpus=all"],
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "forwardPorts": [8888],
  "postCreateCommand": "pip install jupyter matplotlib ipykernel",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-toolsai.jupyter"]
    }
  }
}
```

#### 容器內額外安裝與中文字型

進入 Dev Container 後，需手動安裝專案所需的套件，並設定中文字型供 `matplotlib` 繪圖使用。

```bash
# 1. 補裝 Python 套件
cd /workspace
pip install seaborn imbalanced-learn opencv-python

# 2. 安裝 Noto Sans CJK 中文字型
apt-get update
apt-get install -y fonts-noto-cjk
```

### 3\. 最後的訓練方式 (Docker 模式)

在已進入 Dev Container 的 VS Code 終端：

```bash
cd /workspace/src/semester-project
# 執行訓練腳本
python 07_train_traditional_model.py # 或其他訓練腳本
```