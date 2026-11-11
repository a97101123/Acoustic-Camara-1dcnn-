# 人工特徵、MIC+ACC 拼接、單一 CNN

分類沿用 training wear tertiles：初期、正常、嚴重。先以實體 ToolID
分割 train/validation/test，再決定磨耗門檻。J/R² 選擇、median/IQR
標準化僅使用 train。本版本只支援此 chunk 流程。

## 分段點數由 NFFT 算出

每筆固定約 0.5 秒，並非所有 NFFT 都固定 750 點。

```
window_samples = NFFT
overlap_samples = floor(NFFT × overlap_fraction)
hop_samples = NFFT − overlap_samples
chunk_frames = floor(chunk_seconds × feature_sample_rate / hop_samples)
```

48 kHz、75% overlap、0.5 秒：

| NFFT | hop | 每筆特徵點數 |
| --- | --- | --- |
| 128 | 32 | 750 |
| 192 | 48 | 500 |
| 256 | 64 | 375 |
| 320 | 80 | 300 |
| 384 | 96 | 250 |
| 448 | 112 | 214 |
| 512 | 128 | 187 |

向下取整後的實際秒數寫入 metadata 和 sample_counts.csv。
預設先用 polyphase resampling 統一到 48 kHz，再算特徵。實際車削 MIC
來源為 192 kHz、ACC 為 48 kHz；若不統一，NFFT128 的每筆點數分別為
3000 與 750，不能直接按相同時間點拼接。來源與處理後取樣率皆寫入
sample_counts.csv。`target_sample_rate: null` 可保留原取樣率，但融合要求一致。
每筆輸入是 `[selected_feature, chunk_frames]`。`chunk_retention: all_complete`
保留所有完整 chunk，只捨棄不足一筆的尾端。舊 block_size 與 legacy retention 設定會被拒絕。
每筆保存 ToolID、RecordID、UnitID。既有特徵檔不會補回 chunk，須另建新 Feature Set。

## 特徵

MIC：7 種時域特徵＋短時 FFT power bins（含 DC、不含 Nyquist）。
ACC：各軸 7 種時域特徵＋analysis 原碼的 5 個頻帶最大功率。
使用 periodic Hann；可依每錄音的均值限制頻域峰值。
時域訊號去均值、peak normalization；峰度使用 Pearson 定義。

先對每錄音的各實體段落去頭尾 20%，再串接、計算特徵。
車削單次錄音預設 `segments_per_record=1`；鑽削五孔錄音改為 5。
融合要求 MIC、ACC 的 ToolID/RecordID/chunk ID、標籤、磨耗及取樣設定一致。

這不是 MATLAB 逐數值複製：原碼的頻域先 STFT 再裁五孔，時域先裁五孔
再取窗，本模式先裁再取窗以對齊融合；不做 STFT 邊界 padding。
原碼 ACC 僅頻帶，本模式也提供 ACC 時域特徵；標籤仍採訓練集三分位數。
因此只有在資料、裁切與取樣細節一致時，筆數才可能一致。

## 執行與驗證

執行方式見 README.md。Builder 先產生不含 split/label 的 Feature Set；Trainer 才切分刀具並計算標籤門檻。

J/R² 評分使用每個 ToolID+RecordID 的特徵代表值；CNN 使用保留的原始 chunk 序列。
同位置跨刀平均只用於趨勢圖，不自動用於正式選特徵。圖中的 R² 與逐筆錄音 R² 分開記錄。

sample_counts.csv 記錄各錄音的完整 chunk 數、保留數與不足一筆的尾端特徵點。
來源資料、特徵、split、模型及評估輸出由各自 manifest 記錄。
