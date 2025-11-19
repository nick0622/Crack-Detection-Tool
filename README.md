# 裂縫檢測工具 🔧

使用 YOLOv8 和 Faster R-CNN 的裂縫檢測系統，專為分析機場路面影像設計。

## 概述 📋

本工具使用訓練好的模型自動檢測和分類機場路面影像中的不同類型裂縫。整個過程分為三個簡單步驟：設定環境、預處理影像，然後執行檢測。

## 可用模型 🤖

提供三種訓練好的模型供不同使用情境選擇：

* **YOLOv8 單類別 (Single Class)** 🎯
  * 將裂縫視為單一類別進行檢測
  * 最快的推理速度
  * 適合二元裂縫/無裂縫檢測

* **YOLOv8 四類別 (4 Classes)** 🎯
  * 分類 4 種類型的裂縫
  * 速度與準確度兼顧

* **Faster R-CNN** 🎯
  * 分類 4 種類型的裂縫
  * 準確度較高，推理速度較慢

## 裂縫類型 🔍

多類別模型可檢測並分類 4 種裂縫類型：

* **Transverse** - 垂直於交通方向的裂縫
* **Longitudinal** - 平行於交通方向的裂縫
* **Joint** - 施工接縫處的裂縫
* **Alligator** - 網狀互連裂縫

## 功能特色 ✨

* **多種模型選擇**：可選擇 YOLOv8（單/多類別）或 Faster R-CNN
* **多裂縫檢測**：檢測並分類 4 種裂縫類型
* **影像增強**：可選的預處理功能以改善檢測效果
* **測試時增強 (TTA)**：透過多次預測提高準確度
* **分類別信心度設定**：為每種裂縫類型設定不同的信心度閾值
* **批次處理**：一次處理多張影像
* **詳細結果**：LabelMe 格式 JSON 資料 + 標註影像

## 快速開始 🚀

依照以下步驟設定工具並執行裂縫檢測。

### 1. 安裝相依套件

執行 `setup.bat` 腳本自動安裝所有必要的 Python 套件。

```bash
setup.bat
```

這將會：
- 檢查 Python 安裝
- 安裝必要套件 (numpy, opencv-python, pillow, onnxruntime)
- 建立必要資料夾
- 測試套件匯入

### 3. 預處理影像（可選）

此步驟會對影像進行增強處理，以改善低品質影像的檢測效果。強烈建議用於有雜訊或光線不佳的照片。

執行 `run_preprocessing.bat` 並將影像或資料夾拖曳到視窗中。

增強後的影像會儲存到 `enhanced_images/` 資料夾。

### 4. 執行裂縫檢測

執行 `run_crack_detector.bat` 啟動互動式檢測介面。

功能包括：
- 選擇模型（YOLOv8 單/多類別或 Faster R-CNN）
- 處理單張影像或整個資料夾
- 調整信心度閾值
- 啟用/停用影像增強
- 啟用/停用測試時增強 (TTA)
- 設定分類別信心度閾值

### 5. 裂縫偵測信心閾值
**建議閾值：**

| 裂縫類型 | 預設值 | 建議範圍 | 備註 |
| -------- | ------ | -------- | ---- |
| 橫向裂縫 (Transverse, 0) | 0.15 | 0.15 - 0.30 | 常見，容易檢測 |
| 縱向裂縫 (Longitudinal, 1) | 0.15 | 0.15 - 0.30 | 常見，容易檢測 |
| 接縫裂縫 (Joint, 2) | 0.15 | 0.15 - 0.25 | 可能需要較高閾值以減少誤判 |
| 龜裂 (Alligator, 3) | 0.15 | 0.20 - 0.40 | 複雜圖案，可能需要較高閾值 |

**使用技巧：**
- 較低閾值 = 更多檢測（較多誤判）
- 較高閾值 = 較少檢測（較少誤判，但可能遺漏某些裂縫）
- 根據您的具體需求和影像品質進行調整

## 輸出格式 📤

檢測結果以兩種格式儲存：

### 1. LabelMe JSON 格式

```json
{
  "version": "5.0.1",
  "shapes": [
    {
      "label": "transverse",
      "points": [[x1, y1], [x2, y2]],
      "shape_type": "rectangle",
      "description": "Confidence: 0.85"
    }
  ],
  "imagePath": "image.jpg",
  "imageHeight": 2500,
  "imageWidth": 2500
}
```

**可使用以下工具開啟：**
- [LabelMe](https://github.com/wkentaro/labelme) 標註工具
- 任何 JSON 查看器
- 自訂腳本進行進一步分析

### 2. 標註影像

視覺化呈現包含：
- 彩色邊界框（每種裂縫類型不同顏色）
- 類別標籤及信心度分數
- 與輸入相同的高解析度輸出

## 檔案結構 📂

```
yolo_detector/
├── model/
│   ├── yolov8_single.onnx          # YOLOv8 單類別模型
│   ├── yolov8_multi.onnx           # YOLOv8 四類別模型
│   ├── end2end.onnx                # Faster R-CNN 模型
│   └── classes.json                # 類別名稱（可選）
├── images/
│   └── （您的測試影像）
├── enhanced_images/
│   └── （預處理後的增強影像）
├── results/
│   ├── *.json                      # LabelMe 格式檢測資料
│   └── *.jpg                       # 標註影像
├── inference_yolo.py               # YOLOv8 檢測腳本
├── inference_frcnn.py              # Faster R-CNN 檢測腳本
├── preprocess.py                   # 影像增強工具
├── setup.bat                       # 相依套件安裝
├── run_preprocessing.bat           # 影像預處理介面
├── run_crack_detector.bat          # 主要檢測介面
└── requirements.txt                # Python 相依套件
```

## 系統需求 📦

- Python 3.7+
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0
- onnxruntime >= 1.12.0

所有相依套件會由 `setup.bat` 自動安裝。

## 模型效能比較 ⚡

| 模型 | 速度 | 準確度 | 類別數 | TTA 支援 |
| ---- | ---- | ------ | ------ | -------- |
| YOLOv8 單類別 | ⚡⚡⚡ 快 | ⭐⭐⭐ 良好 | 1 | ✅ 是 |
| YOLOv8 四類別 | ⚡⚡⚡ 快 | ⭐⭐⭐⭐ 非常好 | 4 | ✅ 是 |
| Faster R-CNN | ⚡⚡ 中等 | ⭐⭐⭐⭐⭐ 優秀 | 4 | ✅ 是 |