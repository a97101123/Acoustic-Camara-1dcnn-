# 用 config 切換實驗

## 兩套二分類 R² Top 22

以下四份設定明確使用 Low／High 二分類、argmax、嚴重類索引 1；訓練刀具 1–6、驗證 7/8、測試 9。各跑 6 種 NFFT、seed 42–44、100 epochs。

- `configs/turning_binary_st13_noac.yaml`、`turning_binary_st13_withac.yaml`：St1–6 對 St8–13，排除 St7，其餘不納入。
- `configs/turning_binary_st26_noac.yaml`、`turning_binary_st26_withac.yaml`：St1–12 對 St15–26，排除 St13、14。

四組只跑 R² Top 22，不重跑 J；沿用既有特徵，以設定重新分組及切分。批次啟動前已核對實際來源各 108／216 筆、標籤只有 0/1，並固定設定快照於 `.audit/ml-binary-20261102/`。訓練完成還會核對輸出 metrics 的類別數為 2。

## R² Top 22

`turning_noac_excel_training.yaml`、`turning_withac_excel_training.yaml` 現在引用 `turning_top_training.yaml`。
R² 明確使用 `mode: top`，由 `configs/parameters/turning_top_training/selection.yaml` 控制配額：ACC X/Y/Z 各 4/4/3，MIC 頻域 7、時域 4，共 22 個。
各組按訓練資料的 R² 降冪選取，分數相同按原特徵順序；不使用 0.5 門檻，也不是門檻失敗後補特徵。
候選不足會報錯。門檻統計 CSV 仍保留供分析，不控制 Top 選取。
J、seed、epoch、切分與分類設定不變。舊 `turning_training.yaml` 保留原門檻模式，既有結果不改寫；修改後需建立新的訓練結果，不能接續舊門檻模式的 checkpoint。

## 已接好的實際來源

`configs/turning_noac_paired_excel.yaml` 與 `configs/turning_withac_paired_excel.yaml` 已連到 `data/sources/ml_um/`，不需要再改範例路徑。各有 234 筆、9 把刀、26 個物理階段。
每份來源只有 `FileName`、`WearValue`、`WAVcutstart`。磨耗依使用者 2026-10-31 更正，將 current CSV 的 0.01692 換成 16.92 µm，倍率 1000；沒有插值、重排物理階段或改寫原始表。
換算來源及 hash 在該目錄的 `provenance.json`。程式直接讀已換好的 µm，不會再乘一次。
舊 `cnn_final/inputs/turning.xlsx` 與現行來源部分階段對應不同，未用舊表覆蓋現行來源。

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\turning_noac_paired_excel.yaml -ValidateOnly
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\turning_noac_paired_excel.yaml
```

withAC 換成 `turning_withac_paired_excel.yaml`。兩個來源已通過實際 Excel、音訊及裁切驗證；未進行完整特徵建置或訓練。

在本目錄執行。`includes` 載入共用設定，`overrides` 只寫本次差異。
巢狀 mapping 逐欄覆寫，清單整份取代；未知欄位及重複 include 區塊會報錯。
不會寫回共用 YAML。資料路徑仍相對於最外層啟動 YAML。

`configs/experiment_training.yaml` 示範沿用車削設定，只改成 2 epochs、NFFT 128：

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage train -Config .\configs\experiment_training.yaml -FeatureSet '<已完成的 Feature Set 目錄>'
```

`configs/experiment_excel.yaml` 示範 ML 的三欄 Excel 讀法。先將 `input.excel`、`input.sheet` 改成你的來源與工作表，再驗證：

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\experiment_excel.yaml -ValidateOnly
```

移除 `-ValidateOnly` 才會建立特徵。`sheet` 可以是工作表名稱或從 0 開始的索引。
三欄格式只需 `FileName`、`WearValue`、`WAVcutstart`。例如 `knife3/flank_12.wav` 會得到刀具 T03、階段 M012；刀具也接受 tool3、T03、刀具3 資料夾或直接的數字父資料夾。
`WAVcutstart` 為起點秒數；擷取秒數由 `input.duration_seconds` 控制，範例是 25 秒。
`WearValue` 保留原值，既有流程以微米解讀，不自動換算單位。無法辨識或同刀同階段重複時會報錯。
此格式使用 `sensors: [mic]` 與單一 `conditions`；不會把同一檔案當成多個感測器。
既有 MIC/ACC 配對表仍可使用原欄位。音訊相對路徑以 `input.data_root` 為基準，絕對路徑直接使用。

範例中的 `turning_ml.xlsx` 是待指定的來源檔名，程式不會替你產生或覆寫 Excel。既有 CSV 設定與原始檔未替換。
鑽削可將 includes 改成 `drilling_features.yaml` 或 `drilling_training.yaml`，再覆寫所需差異。

## MIC 與 ACC 三軸都用三欄 Excel

使用 `configs/experiment_paired_excel.yaml`，在 `input.excel` 分別指定 MIC、ACC X/Y/Z 的 Excel：

```yaml
input:
  excel:
    mic: ../data/sources/mic.xlsx
    acc_x: ../data/sources/acc_x.xlsx
    acc_y: ../data/sources/acc_y.xlsx
    acc_z: ../data/sources/acc_z.xlsx
  sheet: 0
  duration_seconds: 25.0
```

每份仍只有 `FileName`、`WearValue`、`WAVcutstart`。依刀具＋階段配對，不依 Excel 列順序。
階段除檔名 flank12，也接受 `knife3/st12/accx.wav` 或 `knife3/flank12/accx.wav` 的階段資料夾。
三軸必須有相同的刀具＋階段集合、磨耗值及起點；MIC 起點可與 ACC 不同，各用自己的 WAVcutstart。
來源缺漏、重複、磨耗不同或三軸起點不一致會直接報錯，不補值也不丟棄未配對列。
Excel 路徑相對於啟動 YAML，Excel 內的相對音訊路徑以 `input.data_root` 為基準。
每份 Excel 及音訊的 hash 都會保存到 `source_manifest.json`。

```powershell
.\tools\run_architecture.ps1 -Process turning -Stage features -Config .\configs\experiment_paired_excel.yaml -ValidateOnly
```

確認後移除 `-ValidateOnly` 才會建立特徵。範例檔名需換成你的實際 Excel。
