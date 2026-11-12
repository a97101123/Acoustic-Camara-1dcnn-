# Acoustic Camera 1D CNN

以 1D CNN 分析加工聲學訊號與刀具磨耗。本專案包含 MATLAB 實驗流程與 Python Wear CNN。

| 流程 | 用途 | 入口 |
| --- | --- | --- |
| Python Wear CNN | 車削／鑽削、MIC 與 ACC 特徵、二／三分類、R² 與 Fisher J | [Python 安裝與執行](New%20cnn/README.md) |
| MATLAB | WAV 裁切、STFT、階段或磨耗分類、WAV 推論 | [MATLAB 環境與腳本索引](matlab/README.md) |

## 資料夾

```text
New cnn/
├── configs/   YAML 啟動設定與共用參數
├── docs/      環境、取樣、設定與實驗說明
├── src/       Python 套件
├── tests/     自動測試
└── tools/     PowerShell 啟動腳本與資料來源工具
matlab/       MATLAB 腳本與使用說明
```

原始 WAV、實驗 Excel、特徵資料集、模型權重與訓練輸出存放在本機。Python 車削 noAC 的修正版流程、所需 Excel 和報表位置見 [Python README](New%20cnn/README.md)。
