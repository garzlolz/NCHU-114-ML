題目：機器學習商品自動化分類-以大買家電商為例；報告時長：約 7 分鐘（預期 13-15 頁）。全局要求：標題直陳步驟/目的，無修飾；使用數據驅動表述；優先用 PNG；使用繁體中文；背景：WSL2/RTX 5070/Keras3+PyTorch；確保文字內容字體清晰度，圖片中文字清晰度；**重要：投影片標題禁止出現任何 Markdown 符號（如 ##、#、**、\*等），標題應為純文本格式，清晰直白\*\*。
投影片結構：
【第 1 頁】標題（標題/副標題）
【第 2 頁】研究目的（背景/挑戰/解決方案）
【第 3 頁】資料收集（爬蟲爬取大買家電商/8 大分類 32 子類別；大類：(1)米油罐頭泡麵 (2)餅乾零食飲料 (3)奶粉養生保健 (4)沐浴開架保養 (5)餐廚衛浴居家 (6)日用清潔用品 (7)家電/3C 配件 (8)文具休閒服飾；爬蟲欄位：brand/name/price/image_url/description；數據規模：原始 7571 筆、清洗後 7307 筆；關鍵發現：商品描述欄位包含產地、成分等文字資訊但覆蓋不完整，圖片反而成為補充商品資訊的主要視覺載體，因此加強圖像特徵提取成為多模態融合的關鍵驅動力）
【第 4 頁】資料前處理（移除售罄/label encoding 8 大類 → 整數 0-7/80-20 stratified split/最終規模 7307）
【第 5-6 頁】圖像特徵工程（RGB 直方圖 32bins×3=96 維捕捉主色調；HSV 直方圖 16bins×3=48 維；HOG pixels_per_cell(100,100) cells_per_block(2,2) 9 方向=576 維捕捉形狀紋理；合計 720 維；StandardScaler 標準化；[留空圖片區域] [使用圖片：資料分佈圖 或 特徵視覺化]）
【第 7 頁】文本特徵工程（TF-IDF：brand+name max_features=2296、description max_features=2769、ngram 1-3、max_df=0.8、sublinear_tf=True、3-fold 交叉驗證調優；合計文本 5065 維；沒有使用斷詞）
【第 8 頁】傳統 ML 訓練（Random Forest+LR；SMOTE k=3 僅訓練集防資料洩漏；RandomizedSearchCV 超參數優化；3-fold Stratified K-Fold；評估指標：準確率）
【第 9 頁】神經網路架構（基於 84.75% 基線；輸入 5786=2296+2769+720+1；Dense(512)+BatchNorm+ReLU+Dropout(0.4)→Dense(256)+BatchNorm+ReLU+Dropout(0.35)→Dense(128)+BatchNorm+ReLU+Dropout(0.3)→Dense(64)+BatchNorm+ReLU+Dropout(0.25)→Dense(8)+Softmax；BatchNorm 設計目的：穩定訓練/提升收斂/正則化/減低 Covariate Shift；Dropout 遞減配置；[留空圖片區域] [使用圖片：NN 網路架構圖(需生成)]）
【第 10 頁】超參數搜索（Grid Search 1080 組 = 12×9×10；Learning Rate 12 值：[0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.00018, 0.0002, 0.00022, 0.00025, 0.00028, 0.0003, 0.0004]；Batch Size 9 值：[16, 18, 20, 22, 24, 26, 28, 32, 48]；Dropout Config 10 組：[(0.3,0.25,0.2,0.15), (0.35,0.3,0.25,0.2), (0.4,0.35,0.3,0.25)基準, (0.45,0.4,0.35,0.3), (0.5,0.45,0.4,0.35), (0.55,0.5,0.45,0.4), (0.4,0.4,0.4,0.4), (0.3,0.3,0.3,0.3), (0.5,0.3,0.3,0.1), (0.2,0.3,0.4,0.5)]；訓練策略：SMOTE(訓練集)k_neighbors=3、先分驗證集再 SMOTE 防污染、Early Stopping patience=10 min_delta=0.001、ReduceLROnPlateau factor=0.5 patience=5 min_lr=1e-6、Adam、Categorical Crossentropy、最大 100 epoch、無 Label Smoothing；Seed Mining Base Seed=821407 確保再現性；Grid Search 全程耗時/平均每組耗時約 100 秒）
【第 11 頁】RTX 5070 訓練挑戰（硬體：RTX 5070 Blackwell/驅動 591.44/CUDA 13.1/Keras3+PyTorch；問題 1：稀疏矩陣相容性 →.toarray().astype(float32) 轉密集；問題 2：validation_split 限制 →train_test_split 手動劃分；問題 3：jit_compile=True 失敗 → 設 jit_compile=False；問題 4：Label Smoothing 實驗 → 值 0.1 導致準確率下降至 77%、無 Label Smoothing 保持 84.75%→ 決定不採用；單次 NN 訓練耗時約 3 分鐘；斷點續練機制：讀取 keras_grid_search_optimized.csv 避免重複訓練）
【第 12 頁】模型準確率比較（表格：Keras 基線 Seed 821407 84.75%/3.4 分鐘、Random Forest 80.03%/2.1 秒、Logistic Regression 73.6%/14 秒；對比：準確率 vs 訓練時間權衡/複雜度分析；[留空圖片區域 1] [使用圖片：model_comparison.png] [留空圖片區域 2] [使用圖片：training_time_comparison.png] [留空圖片區域 3] [使用圖片：performance_time_tradeoff.png]）
【第 13 頁】集成學習（載入既有已訓練模型：RF pkl + Keras model，不重訓，無訓練時間；策略 1 精細權重搜索：最佳權重 RF=0.77、Keras=0.23，集成準確率 86.05%（相對 Keras +1.30%）；[留空圖片區域] [使用圖片：ensemble_weight_search.png]；策略 2 Stacking：測試集切 50% 訓練 Meta-Model/50% 驗證，Meta-Model=Logistic Regression，準確率 84.40%；[留空圖片區域] [使用圖片：final_optimized_ensemble_matrix.png]）
【第 14 頁】查表法基線（方法：8 大類各 10-30 關鍵詞、逐詞匹配品牌+商品名+描述、計分取 argmax；統計：未分類 3611 筆、已分類準確率 63%、整體準確率 32%；對比：查表優點可解釋/無訓練/推理快、局限覆蓋率低/無泛化/維護複雜；ML 優勢自動特徵/泛化強/適應新數據、性能提升 ML vs 查表法+50%；[留空圖片區域] [使用圖片：lookup_confusion_matrix.png]）
【第 15 頁】單筆商品預測展示（使用 11_predict_single_product：載入 output/processed_features.pkl 的 tfidf_name、tfidf_desc、scaler_img、scaler_price、label_encoder，並載入最佳模型；特徵組合與訓練一致：TF-IDF(2296+2769)+圖像(720)+價格(1)=5786；提供 2 個示例：示例 1 含圖片路徑+價格、示例 2 無圖片路徑；輸出：預測類別、confidence、Top-k 機率條形文字；並支援互動輸入與 CSV 批次預測輸出 output/predictions_result.csv，統計平均 confidence、各類別數量、confidence 區間分佈(>0.90、0.60-0.90、<0.60)；[留空圖片或終端輸出區域] [使用圖片或截圖：11_predict_single_product 執行結果]）
【第 16 頁】結論（核心發現：多模態>單模態、BatchNorm+Dropout 穩定性、1080 組搜索發現最優；選定：Ensemble(加權投票 RF0.77+Keras0.23) 86.05% 為最佳；創新點：Seed Mining Base Seed 821407 穩定可重現、Label Smoothing 實驗證實不適用(77%)、RTX50 系列相容性優化、斷點續練機制減少重複計算；限制與展望：查表法準確率低）
【第 17 頁】未來改進（文本升級：BERT 取代 TF-IDF、上下文語義、預期+3~5%；圖像升級：ViT/ResNet50/EfficientNet 取代 HOG+直方圖、長距離依賴/位置不敏感、預期+2~4%；多模態融合：聯合訓練 BERT 編碼器+ViT 編碼器、交叉注意力；不涉及應用層面）
檢查清單：

- 頁數約 13-17 頁約 7 分鐘(可合併第 16-17 頁以回到 15 頁)
- 所有圖表留空，下方標註要用哪張圖（由用戶自行補上）
- 投影片標題禁止出現 Markdown 符號（## 、 # 、 \*_ 、 _ 等），標題為純文本
- 圖像特徵 720=96+48+576 維
- Grid Search 1080=12×9×10 組完整參數範圍詳列
- RTX50 系列問題與解法
- BatchNorm 設計目的與層數
- Seed Mining Base Seed 821407 無 Label Smoothing 機制
- Label Smoothing 實驗失敗(77%)記錄
- 8 大分類詳列
- 集成最佳權重 RF0.77+Keras0.23 準確率 86.05%
- Stacking 84.40% 設定說明
- 補上 11_predict_single_product 單筆/批次預測展示
- 未來展望專注準確度 BERT+ViT 無應用
- 每頁 30-40 秒
- 商品描述包含產地、成分等資訊但不完整，圖片為主視覺載體的發現正確述明
  需補充數值：
  第 10-11 頁 Grid Search 全程耗時/平均每組耗時；若要呈現單筆推論速度則另行量測
  圖片清單（由用戶自行補上）：
- 第 5-6 頁：資料分佈圖 或 特徵視覺化
- 第 9 頁：NN 網路架構圖(需生成)
- 第 12 頁：ensemble_weight_search.png、final_optimized_ensemble_matrix.png
- 第 13 頁：model_comparison.png、training_time_comparison.png、performance_time_tradeoff.png
- 第 14 頁：lookup_confusion_matrix.png
- 第 15 頁：11_predict_single_product 執行結果

## 2

======================================================================================
題目：機器學習商品自動化分類-以大買家電商為例；報告時長：約 7 分鐘（預期 13-15 頁）。全局要求：標題直陳步驟/目的，無修飾；使用數據驅動表述；優先用 PNG；使用繁體中文；背景：WSL2/RTX 5070/Keras3+PyTorch；確保文字內容字體清晰度，圖片中文字清晰度；投影片標題禁止出現任何 Markdown 符號，標題應為純文本格式，清晰直白。
圖片占位符要求：所有需要補上結果圖的頁面，必須生成明確的圖片占位符區域（使用灰色或淡色矩形框、虛線邊界等），占位符內部標註「[圖片區域]」並下方清晰標註該位置需使用的圖片名稱，確保用戶能清楚看到圖片應該貼放的位置和大小。
投影片結構：
【第 1 頁】標題（標題/副標題）
【第 2 頁】研究目的（背景/挑戰/解決方案）
【第 3 頁】資料收集（爬蟲爬取大買家電商/8 大分類 32 子類別；大類：(1)米油罐頭泡麵 (2)餅乾零食飲料 (3)奶粉養生保健 (4)沐浴開架保養 (5)餐廚衛浴居家 (6)日用清潔用品 (7)家電/3C 配件 (8)文具休閒服飾；爬蟲欄位：brand/name/price/image_url/description；數據規模：原始 7571 筆、清洗後 7307 筆；關鍵發現：商品描述欄位資訊內容稀少，僅包含產地、成分等簡單基本資訊，因此加強圖像特徵提取成為多模態融合的關鍵驅動力）
【第 4 頁】資料前處理（移除售罄/label encoding 8 大類 → 整數 0-7/80-20 stratified split/最終規模 7307）
【第 5-6 頁】圖像特徵工程（RGB 直方圖 32bins×3=96 維捕捉主色調；HSV 直方圖 16bins×3=48 維；HOG pixels_per_cell(100,100) cells_per_block(2,2) 9 方向=576 維捕捉形狀紋理；合計 720 維；StandardScaler 標準化；
[生成灰色占位符矩形框，寬高約 800x400px]
[圖片區域：資料分佈圖 或 特徵視覺化]）
【第 7 頁】文本特徵工程（TF-IDF：brand+name max_features=2296、description max_features=2769、ngram 1-3、max_df=0.8、sublinear_tf=True、3-fold 交叉驗證調優；合計文本 5065 維；沒有使用斷詞）
【第 8 頁】傳統 ML 訓練（Random Forest+LR；SMOTE k=3 ；RandomizedSearchCV 超參數優化；3-fold Stratified K-Fold；評估指標：準確率）
【第 9 頁】神經網路架構（基於 84.75% 基線；輸入 5786=2296+2769+720+1；Dense(512)+BatchNorm+ReLU+Dropout(0.4)→Dense(256)+BatchNorm+ReLU+Dropout(0.35)→Dense(128)+BatchNorm+ReLU+Dropout(0.3)→Dense(64)+BatchNorm+ReLU+Dropout(0.25)→Dense(8)+Softmax；BatchNorm 設計目的：穩定訓練/提升收斂/正則化/減低 Covariate Shift；Dropout 遞減配置；
[生成灰色占位符矩形框，寬高約 600x450px，中心標註「圖片區域」]
[圖片區域：NN 網路架構圖(需生成)]）
【第 10 頁】超參數搜索（Grid Search 1080 組 = 12×9×10；Learning Rate 12 值：[0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.00018, 0.0002, 0.00022, 0.00025, 0.00028, 0.0003, 0.0004]；Batch Size 9 值：[16, 18, 20, 22, 24, 26, 28, 32, 48]；Dropout Config 10 組：[(0.3,0.25,0.2,0.15), (0.35,0.3,0.25,0.2), (0.4,0.35,0.3,0.25)基準, (0.45,0.4,0.35,0.3), (0.5,0.45,0.4,0.35), (0.55,0.5,0.45,0.4), (0.4,0.4,0.4,0.4), (0.3,0.3,0.3,0.3), (0.5,0.3,0.3,0.1), (0.2,0.3,0.4,0.5)]；訓練策略：SMOTE(訓練集)k_neighbors=3、先分驗證集再 SMOTE 防污染、Early Stopping patience=10 min_delta=0.001、ReduceLROnPlateau factor=0.5 patience=5 min_lr=1e-6、Adam、Categorical Crossentropy、最大 100 epoch、無 Label Smoothing；Seed Mining Base Seed=821407 確保再現性；Grid Search 全程耗時/平均每組耗時約 200 秒）
【第 11 頁】RTX 5070 訓練挑戰（硬體：RTX 5070 Blackwell/驅動 591.44/CUDA 13.1/Keras3+PyTorch；問題 1：稀疏矩陣相容性 →.toarray().astype(float32) 轉密集；問題 2：validation_split 限制 →train_test_split 手動劃分；問題 3：jit_compile=True 失敗 → 設 jit_compile=False；問題 4：Label Smoothing 實驗 → 值 0.1 導致準確率下降至 77%、無 Label Smoothing 保持 84.75%→ 決定不採用；單次 NN 訓練耗時約 3 分鐘；斷點續練機制：讀取 keras_grid_search_optimized.csv 避免重複訓練）
【第 12 頁】模型準確率比較（表格：Keras 基線 Seed 821407 84.75%/3.4 分鐘、Random Forest 80.03%/2.1 秒、Logistic Regression 73.6%/14 秒；對比：準確率 vs 訓練時間權衡/複雜度分析；
[生成灰色占位符矩形框 1，寬高約 600x350px]
[圖片區域 1：model_comparison.png]
[生成灰色占位符矩形框 2，寬高約 600x350px]
[圖片區域 2：training_time_comparison.png]
[生成灰色占位符矩形框 3，寬高約 600x350px]
[圖片區域 3：performance_time_tradeoff.png]）
【第 13 頁】集成學習（載入既有已訓練模型：RF pkl + Keras model，不重訓，無訓練時間；策略 1 精細權重搜索：最佳權重 RF=0.77、Keras=0.23，集成準確率 86.05%（相對 Keras +1.30%）；
[生成灰色占位符矩形框，寬高約 700x380px]
[圖片區域：ensemble_weight_search.png]；策略 2 Stacking：測試集切 50% 訓練 Meta-Model/50% 驗證，Meta-Model=Logistic Regression，準確率 84.40%；
[生成灰色占位符矩形框，寬高約 700x380px]
[圖片區域：final_optimized_ensemble_matrix.png]）
【第 14 頁】查表法基線（方法：8 大類各 10-30 關鍵詞、逐詞匹配品牌+商品名+描述、計分取 argmax；統計：未分類 3611 筆、已分類準確率 63%、整體準確率 32%；對比：查表優點可解釋/無訓練/推理快、局限覆蓋率低/無泛化/維護複雜；ML 優勢自動特徵/泛化強/適應新數據、性能提升 ML vs 查表法+50%；
[生成灰色占位符矩形框，寬高約 700x400px]
[圖片區域：lookup_confusion_matrix.png]）
【第 15 頁】單筆商品預測展示（使用 11_predict_single_product：載入最佳模型；特徵組合與訓練一致：TF-IDF(2296+2769)+圖像(720)+價格(1)=5786；提供 2 個示例：示例含圖片路徑+價格；輸出：預測類別、confidence、Top-k 機率條形文字；並支援互動輸入與 CSV 批次預測輸出 output/predictions_result.csv，統計平均 confidence、各類別數量、confidence 區間分佈(>0.90、0.60-0.90、<0.60)；
[生成灰色占位符矩形框，寬高約 800x350px]
[圖片區域：11_predict_single_product 執行結果]）
【第 16 頁】結論（核心發現：多模態>單模態、BatchNorm+Dropout 穩定性、1080 組搜索發現最優；選定：Ensemble(加權投票 RF0.77+Keras0.23) 86.05% 為最佳；使用 Seed Mining Base Seed 821407 穩定可重現、Label Smoothing 實驗證實不適用(77%)、RTX50 系列相容性優化、斷點續練機制減少重複計算；限制與展望：查表法準確率低）
【第 17 頁】未來改進（文本升級：BERT 取代 TF-IDF、上下文語義、預期+3~5%；圖像升級：ViT/ResNet50/EfficientNet 取代 HOG+直方圖、長距離依賴/位置不敏感、預期+2~4%；多模態融合：聯合訓練 BERT 編碼器+ViT 編碼器、交叉注意力；不涉及應用層面）
檢查清單：

- 頁數約 13-17 頁約 7 分鐘(可合併第 16-17 頁以回到 15 頁)
- 所有圖表留空且生成明確灰色占位符矩形框，下方標註圖片名稱
- 占位符框內標註「圖片區域」便於用戶識別
- 投影片標題禁止出現 Markdown 符號（## 、 # 、 \*_ 、 _ 等），標題為純文本
- 圖像特徵 720=96+48+576 維
- Grid Search 1080=12×9×10 組完整參數範圍詳列
- RTX50 系列問題與解法
- BatchNorm 設計目的與層數
- Seed Mining Base Seed 821407 無 Label Smoothing 機制
- Label Smoothing 實驗失敗(77%)記錄
- 8 大分類詳列
- 集成最佳權重 RF0.77+Keras0.23 準確率 86.05%
- Stacking 84.40% 設定說明
- 補上 11_predict_single_product 單筆/批次預測展示
- 未來展望專注準確度 BERT+ViT 無應用
- 每頁 30-40 秒
- 商品描述資訊稀少（僅產地、成分等基本資訊），圖片為主視覺載體的發現正確述明
  需補充數值：
  第 10-11 頁 Grid Search 全程耗時/平均每組耗時；若要呈現單筆推論速度則另行量測
  圖片清單（由用戶自行補上）：
- 第 5-6 頁：資料分佈圖 或 特徵視覺化
- 第 9 頁：NN 網路架構圖(需生成)
- 第 12 頁：model_comparison.png、training_time_comparison.png、performance_time_tradeoff.png
- 第 13 頁：ensemble_weight_search.png、final_optimized_ensemble_matrix.png
- 第 14 頁：lookup_confusion_matrix.png
- 第 15 頁：11_predict_single_product 執行結果
