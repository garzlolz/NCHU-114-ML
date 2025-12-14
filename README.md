# NCHU 114, 資管系在職碩專班 專題與功課

## 專題目錄
```text
. (專案根目錄)
├── README.md                                               # 專案說明文件
└── src/                                                    # 來源碼 (Source) 資料夾
    ├── semester-project/                                   # 專題程式存放位置
    │   │── output/                                         # 輸出
    │   │   ├── images/                                     # 爬蟲圖檔 
    │   │   ├── result_images/                              # matplotlib 圖檔
    │   │   ├── savesafe_cleaned_products_20251207_143149   # 前處理後資料 
    │   │   └──  savesafe_products_20251207_143149          # 大買家爬蟲資料
    │   └── 08_winpytorch_train_neural_network.py           # (Keras 3 + PyTorch 版本)
    └── week...                                             # 課程功課
````

-----

## 一、環境目標

本專案主要在 Windows 11 + WSL2 + RTX 5070 硬體環境下執行。由於 RTX 50 系列 (Blackwell 架構) 較新，舊版 TensorFlow 支援度不佳，因此本文件記錄了兩種可行的執行方案：
抱歉，是我記錯了。您說得對，既然您的 Host 端 `nvidia-smi` 顯示的是 **CUDA 13.1**，且為了讓 RTX 5070 發揮最新架構（Blackwell）的完整效能，應該選用 **CUDA 13.0** 的版本（通常 PyTorch 會有對應大版本的 build，例如 `cu130`）。

先前的 `cu124` 是保守的穩定版選項，但針對您的硬體與之前的指令，我們應該直上 **CUDA 13.0**。

以下是修正後的 **方案 A** 區塊，請使用這段取代原本的內容：

-----

## 二、方案 A (推薦)：本機 Conda 環境 (PyTorch + Keras 3)

鑑於 TensorFlow 對 RTX 50 系列的原生支援尚不完整，改用 **PyTorch** 作為 Keras 3 的後端引擎是目前最穩定且效能最佳的方案。此方法直接在 Windows/WSL 的 Conda 環境執行，無需透過 Docker。

### 1\. 建立 Conda 虛擬環境

建立一個 Python 3.12 的乾淨環境（建議命名為 `py312_keras_torch`）：

```bash
conda create -n py312_keras_torch python=3.12 -y
conda activate py312_keras_torch
```

### 2\. 安裝 PyTorch (CUDA 13.0 版本)

您的 `nvidia-smi` 顯示驅動程式支援至 **CUDA 13.1**。為了完全發揮 RTX 5070 (Blackwell 架構) 的效能，請安裝 **CUDA 13.0** 對應版本的 PyTorch：

```bash
# 指名下載 cu130 (CUDA 13.0) 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

> **注意：** 若 `cu130` 尚未在 stable 通道釋出，可改用 nightly build (開發版) 以取得最新支援：
> `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130`

### 3\. 安裝 Keras 3 與其他相依套件

安裝最新版 Keras (預設為 Keras 3) 以及專案所需的資料處理套件：

```bash
# 安裝 Keras 與常用工具
pip install keras numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# 安裝 OpenCV (若專案有影像處理需求)
pip install opencv-python
```

### 4\. 設定 Keras 後端

在 Python 程式碼 (`.py`) 的最上方加入環境變數設定，告訴 Keras 使用 PyTorch 作為運算核心：

```python
import os
# 必須在 import keras 之前設定
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"
import keras
```

### 5\. 驗證安裝

執行以下指令確認 PyTorch 是否抓得到 RTX 5070 且 CUDA 版本正確：

```bash
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**預期輸出：**

  - `CUDA Available: True`
  - `CUDA Version: 13.0` (或接近版本)
  - `Device: NVIDIA GeForce RTX 5070`

### 6\. 執行訓練

進入專案目錄並執行對應的訓練腳本：

```bash
cd src/semester-project
python 08_winpytorch_train_neural_network.py
```

-----

## 三、方案 B (備用)：NVIDIA Docker 容器

若必須使用 TensorFlow 原生環境，建議使用此 Docker 方案。

### 1\. 安裝與測試步驟

1.  **確認 WSL 看得到 GPU**

    ```bash
    nvidia-smi
    ```

    看到 RTX 5070、Driver 591.44、CUDA Version 13.1。

2.  **測試 Docker + GPU**

    ```bash
    docker run --rm --gpus all nvidia/cuda:12.8.0-devel-ubuntu22.04 nvidia-smi
    ```

    容器內同樣顯示 RTX 5070，代表 Docker GPU passthrough 正常。

3.  **測試 NVIDIA TensorFlow 容器**

    ```bash
    docker run -it --rm --gpus all nvcr.io/nvidia/tensorflow:25.01-tf2-py3 \
      python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
    ```

    顯示 TensorFlow 2.17.0，且偵測到 `/physical_device:GPU:0`，證明容器內 TF 可以用 RTX 5070。

### 2\. 本機 pip 版 TensorFlow 失敗原因紀錄

在 WSL conda 環境 `wsl_ml_hw` 中用 pip 安裝了 `tensorflow==2.20.0` 和一堆 `nvidia-*-cu12` 套件，訓練時出現：

  - `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0`
  - `CUDA_ERROR_INVALID_PTX`、`CUDA_ERROR_INVALID_HANDLE`

原因是 pip 版 TF 2.20 沒有為 Blackwell 的 compute capability 12.0 編好對應的 CUDA binary，對 RTX 50 系列在新 driver 上相容性不好。
解法是改用 **NVIDIA 官方 TensorFlow 容器**，裡面已針對新架構編譯好。

### 3\. 改用 VS Code Dev Container 的設定

1.  在專案根目錄建立 `.devcontainer/devcontainer.json`：

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
          "extensions": [
            "ms-python.python",
            "ms-toolsai.jupyter"
          ]
        }
      }
    }
    ```

    這會讓 VS Code 把目前專案資料夾掛載到容器的 `/workspace`，並自動開啟有 GPU 的 TensorFlow 開發環境。

2.  在 VS Code 中：

      - `Ctrl + Shift + P` → `Dev Containers: Reopen in Container`
      - 進入容器後終端提示變成 `root@...:/workspace#`，`python -c "import tensorflow as tf; ..."` 可看到 GPU。

### 4\. 在容器裡補裝需要的套件與字型

1.  **補裝 Python 套件**（因為容器和本機環境獨立）：

    ```bash
    cd /workspace
    pip install seaborn imbalanced-learn opencv-python
    ```

    其他如 pandas、matplotlib、scikit-learn 在 image 裡已預裝。

2.  **安裝中文字型 Noto Sans CJK**（給 matplotlib 用）：

    ```bash
    apt-get update
    apt-get install -y fonts-noto-cjk
    ls /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
    ```

    程式裡 `FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"` 就能正常載入。

### 5\. 最後的訓練方式 (Docker 模式)

在已進入 Dev Container 的 VS Code 終端：

```bash
cd /workspace/src/semester-project
python 07_train_model.py
```

這樣：

  - 程式碼仍在主機 `~/repos/NCHU-114-Master-ML-homework` 裡（透過掛載）。
  - 訓練在容器環境中進行，使用 NVIDIA 官方編譯好的 TensorFlow + CUDA 12.8，穩定支援 RTX 5070（Blackwell）。
  - 不再依賴本機 conda 的 TensorFlow，避免 `CUDA_ERROR_INVALID_PTX` 類問題。


```