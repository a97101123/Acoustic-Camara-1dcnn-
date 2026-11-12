# 用 YAML 控制分類

在此 `New cnn` 目錄執行 PowerShell。程式維持 PyTorch 與現有完整 chunk 特徵流程。

設定按功能拆分。`configs/*_features.yaml`、`*_training.yaml` 是啟動清單，只有 `includes`，不放實驗參數。實際參數在 `configs/parameters/<清單名稱>/`。

| 檔案 | 控制項目 |
| --- | --- |
| dataset.yaml | 來源、感測器或刀具、stage、標籤、實驗篩選 |
| features.yaml | NFFT、取樣率、裁切、重疊、chunk、ACC 頻帶 |
| model.yaml | CNN filters、kernel、dropout、正規化及 median/IQR scaling |
| training.yaml | seed、epoch、batch、學習率、Adam、裝置、執行緒 |
| evaluation.yaml | 切分、驗證規則、推論裁切及批次大小 |
| selection.yaml | J 配額、R² 門檻與門檻報表 |
| output.yaml | 輸出位置、圖表 DPI、HDF5 I/O |

缺少必要欄位、未知欄位、重複 YAML key、重複功能區塊、循環引用都會報錯，不套用 Python 內建實驗參數。功能檔內資料與輸出相對路徑以啟動清單所在目錄解析；`includes` 路徑以包含它的 YAML 所在目錄解析。`null` 可明確表示不篩選刀具、NFFT 或條件。

## St1–13 二分類，排除 St7

由 `configs/turning_binary_training.yaml` 啟動，分別編輯 `configs/parameters/turning_binary_training/dataset.yaml`、`training.yaml`、`evaluation.yaml`。以下示意各功能檔中的對應區塊，不要合併回啟動清單。

```yaml
dataset:
  knife_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  stage_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  exclude_stages: [7]
  labeling:
    method: stage_groups
    groups:
      - [1, 2, 3, 4, 5, 6]
      - [8, 9, 10, 11, 12, 13]
  class_names: [Low, High]
training:
  severe_class_index: 1
evaluation:
  strategy: fixed_knife
  test_knife: 9
  val_knives: [7, 8]
  prediction_rule: argmax
```

範例將刀具 1–6 訓練、7/8 驗證、9 測試。未分到驗證或測試的已納入刀具均為訓練，因此要限定 1–9 就保留 knife_ids。數字 9 對應來源 T09，也可明確寫 ToolID 字串。未知、重複或重疊的刀具設定會被拒絕。

Stage 對應車削來源 RecordID 的 `M001`、`M002` 等編號。啟用 stage 篩選時不接受其他命名方式，也不將鑽削孔號猜成 Stage。groups 不可重疊。只保留 groups 聯集，再套用 stage_ids 和 exclude_stages。每個 split 必須包含全部類別。

`severe_class_index` 是從 0 開始的正類索引。二分類 [Low, High] 的 High 是 1。類別顯示名稱可以修改，CSV 使用穩定的 `p_class_0` 與 `p_severe` 欄位。

既有已完成且通過 hash／all_complete 檢查的 Feature Set 可以直接用新標籤訓練，不必重抽音訊。

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage train -Config .\configs\turning_binary_training.yaml -FeatureSet '你的已完成 Feature Set 目錄'
```

要從原始來源建立特徵後訓練，可用單一工作流程入口。這會建立新輸出。

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv-cuda\Scripts\python.exe -u -m wear_cnn.cli run-workflow --config .\configs\turning_binary_workflow.yaml
```

CPU 環境將 `.venv-cuda` 改成 `.venv-review`。工作流程引用的兩份 YAML 分別控制音訊特徵與訓練；本次只改分類設定時可沿用相同特徵。

## 三分類

若要指定三個階段群組，將 groups 改成三組、class_names 改成三個名稱，並設定 severe_class_index 為嚴重類所在索引。無須修改 Python 或另填 num_classes。stage_ids／exclude_stages 也要涵蓋你希望保留的階段。

若要保留原有按磨耗的三分類，使用 `turning_training.yaml`。

```yaml
dataset:
  labeling:
    method: training_tertiles
  class_names: [初期, 正常, 嚴重]
training:
  severe_class_index: 2
evaluation:
  strategy: random_tools
  prediction_rule: validation_threshold
```

training_tertiles 只用訓練 record 的磨耗三分位數決定三分類標籤。random_tools 使用 evaluation.yaml 明寫的比例和最少刀具數做整刀切分。validation_threshold 只用驗證資料挑正類門檻，argmax 直接取最大機率。所有欄位必須明確提供。

## 其他設定與輸出

- 特徵 YAML 控制輸入來源、感測器、取樣、chunk、NFFT 與輸出位置。
- 訓練 YAML 的 nffts／conditions 可以從已完成 Feature Set 篩選實驗。
- training 的 runs、seed_start、epochs、batch_size、validation_batch_size、checkpoint_every 控制訓練。模型架構參數在 model.yaml；輸出類別數由分類設定決定。
- selection 沿用本專案的 J 軸／時頻配額與 R² threshold。二分類範例沿用原有特徵規則，並不自動轉換成舊 TensorFlow 專案的 top14 或 ACC7/MIC7 規則。
- history CSV、學習曲線、模型、逐筆機率及報表按原本目錄輸出，混淆矩陣會依分類數產生 2×2 或 3×3。

分類、階段群組與切分設定會寫入模型及部署資料。訓練與推論使用相同分類規則。修改設定後建立新 run；無法用不相容設定續跑舊 checkpoint。

完整車削／鑽削批次使用 `tools/run_four_architecture.ps1 -Config configs/four_architecture.yaml`。批次清單引用 `parameters/batch/jobs.yaml`，明列各流程的特徵、訓練清單。工作數由設定計算，不固定為 56。原有條件 × NFFT × J/R² × seeds 網格保留。

`configs/h_drive/` 是舊 CLI 的歷史設定，不屬於目前入口。舊模型缺少明確架構與分類 metadata 時不會猜測相容參數，請用目前清單另建新 run。

評估仍以 chunk 為樣本，與舊專案逐 record 的準確率不可直接視為同一實驗。
