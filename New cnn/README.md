# Wear CNN

車削／鑽削刀具磨耗分類。YAML 可控制二／三分類、階段分組、刀具切分及預測規則。正式流程只有建立 Feature Set、訓練與報告；原始資料和既有結果不會自動覆寫。

## 執行

在專案目錄使用 PowerShell：

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\turning_features.yaml -ValidateOnly
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\turning_features.yaml
.\tools\run_architecture.ps1 -Process turning -Stage train -Config .\configs\turning_training.yaml -FeatureSet '<建立完成的 Feature Set 目錄>'
```

鑽削將 Process 改為 drilling。車削與鑽削全部依序執行：

```powershell
.\tools\run_four_architecture.ps1 -Config .\configs\four_architecture.yaml
```

此命令會重建全部特徵並訓練兩流程、四條件、七種 NFFT、J/R²、每組三個 seed。失敗會記錄，沒有特徵通過 R² 門檻時不補低分特徵。

## 設定

共用設定加實驗差異、直接讀 Excel 工作表的範例見 [EXPERIMENTS.md](EXPERIMENTS.md)。

使用時明確選擇 turning_features.yaml、turning_training.yaml、drilling_features.yaml、drilling_training.yaml。這些檔案只有功能設定引用，參數拆在 `configs/parameters/<名稱>/`，缺值或未知欄位直接報錯。turning_binary_training.yaml 提供 St1–13 排除 St7 的二分類範例；turning_binary_workflow.yaml 串接特徵建立與二分類訓練。

- features 設定來源、感測器、裁切、取樣與完整 chunk 保留。
- training 設定刀具切分、J/R²、seed、epoch 和裝置。
- 新實驗保留全部完整 chunk。block_size 與舊 chunk 保留規則已移除。
- 同刀只屬於一個 split；標籤三分位數、特徵選擇及標準化只使用訓練資料。
- 每筆加工的 chunk 共用來源 WearValue。J 使用三類標籤；R² 使用逐筆訓練錄音代表值，不會自動採用試跑的跨刀平均方案。
- 嚴重類機率門檻仍以驗證 Macro-F1 選擇，模型與分類規則未改。

二分類設定與功能檔位置見 [CONFIGURATION.md](CONFIGURATION.md)。三分類清單明確指定訓練磨耗三分位數、隨機整刀切分及驗證門檻。二分類範例使用 argmax；類別數由 groups 與 class_names 決定，模型、機率欄位、混淆矩陣及 WAV 推論自動一致。

目前輸入是 data/sources/turning_current.csv、drilling_current.csv。車削新表由完整的 h_drive 來源另建；鑽削沿用三階段來源及每段五孔切分。原始表及 WearValue 未改，來源 hash 記錄於同名 provenance.json。

在新工作環境先建立來源表，輸出已存在時會拒絕覆寫：

```powershell
.\.venv-review\Scripts\python.exe tools/build_current_sources.py --process turning --data-root D:/turning
.\.venv-review\Scripts\python.exe tools/build_current_sources.py --process drilling --data-root D:/drilling
```

舊 CLI、舊 record/fusion 讀取相容層與舊設定已移除。舊檔保留在磁碟，但本版本要求重新建立 Feature Set 並訓練。舊程式可從本地 Git 的清理前版本取得。

## 檢查

```powershell
.\.venv-review\Scripts\python.exe -m pytest -q
.\.venv-review\Scripts\ruff.exe check src tools tests
.\.venv-review\Scripts\ruff.exe format --check src tools tests
```

其他說明見 ARCHITECTURE_RUNBOOK.md、ANALYSIS_SAMPLING.md、PYTHON_ENVIRONMENT.md。
