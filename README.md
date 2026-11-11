# Acoustic Camera 1D CNN

使用 MATLAB 處理聲學錄音，並以 1D CNN 分類加工階段或刀具磨耗。原始碼與設定集中在 [matlab/](matlab/)。

## 執行環境與本機資料

使用具備 Signal Processing Toolbox 和 Deep Learning Toolbox 的 MATLAB。在 MATLAB 將目前資料夾切換到 `matlab/`，讓腳本從該目錄解析相對路徑。

- `acoustic/<條件>/` 放原始 WAV 與裁切後的 WAV。
- `file_name_<標籤>.xlsx` 和 `mic_list_<標籤>.xlsx` 放在 MATLAB 目前資料夾。
- `wavecut/mic_wave_cut_<標籤>.xlsx` 提供 WAV 裁切資訊。
- `flank wear.xlsx` 提供磨耗分類所需的量測資料。可在 `tw_cfg.m` 覆寫檔名、資料列、單位比例及分類門檻。
- 腳本使用 `03m0d`、`06m0d` 和 `06m15d` 作為資料集標籤。準備其他資料集時，請依實驗配置調整刀具數、階段數及每段孔數。

## 腳本索引

| 用途 | MATLAB 檔案 |
| --- | --- |
| 取樣率、STFT、實驗配置與路徑 | `tw_cfg.m` |
| WAV 裁切與特徵擷取 | `tw_wave_cut_mic.m`、`tw_mic_traincellgen_freq.m` |
| 磨耗標籤、訓練資料與整刀交叉驗證 | `tw_build_wear_map.m`、`tw_train_pt.m`、`tw_train_pt_kfold.m` |
| 1D CNN 訓練 | `tw_train_1dcnn.m` |
| 批次入口 | `oneclick.m`（階段分類）、`oneclick_waer.m`（磨耗分類，使用已裁切的 WAV） |
| WAV 推論 | `wavprec.m`（選取 WAV）、`tw_predict_wav.m`（執行推論） |

批次入口會為每個標籤準備訓練資料，迴圈結束後用最後一組設定呼叫 CNN 訓練。訓練產物預設位於 `cnn2/train_ready2/`。推論前請在 `wavprec.m` 指定實際的模型檔案。
