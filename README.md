# BTC 策略回測 + AI 訊號 + 儀表板

本專案提供一套端到端流程：
- 抓取 BTC K 線歷史資料（Binance）
- 增量更新（尾段重抓 + 去重）
- 特徵工程（MA、MACD、RSI、布林、ATR、量能壓力、支撐/壓力、趨勢、波動、回撤）
- AI 訓練（多/空/觀望分類 + 槓桿建議回歸）
- 回測（含交易成本、回撤停損）
- 儀表板（多週期切換 + SNR 線 + 指標視覺化）
- OKX 永續合約（SWAP）模擬盤交易（paper）

## 為什麼增量更新省效能
不需要每次重抓全歷史，做法是：
1. 保留舊資料
2. 只重抓最近一段重疊窗口（避免漏資料/修正）
3. 用 K 線開盤時間去重合併

## 安裝
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 設定
把 `.env.example` 複製成 `.env`，依需求修改參數。

## 執行（命令列）
增量更新 + 訓練 + 回測：
```bash
python run.py
```

強制重抓全歷史：
```bash
python run.py --full-refresh
```

## 儀表板（建議）
啟動 Streamlit 儀表板：
```bash
streamlit run dashboard.py
```

儀表板功能：
- 多週期切換：`5m / 15m / 30m / 1h / 1d`
- 一鍵：全歷史重抓 / 快速更新 / 增量更新+重訓回測
- K 線圖 + MA12/MA24
- SNR（支撐/壓力）多週期重疊水平線，標籤例如 `S 5m,1h,1d`（標線會從該價位首次觸及的 K 線開始繪製）
- 趨勢高低點、MACD、ATR、買賣機率橫條圖、恐懼貪婪儀表板等

### 目前 Dashboard 需求（2026-05-20）
- 頂部 Banner 顯示：`OKX 餘額 / 最新收盤價 / 看漲機率 / 看跌機率 / 執行槓桿`
- `黑天鵝緩衝金` 與 `黑天鵝觸發門檻` 置於頂部第二列控制區，避免擁擠
- 只保留「純 AI 自動交易」，移除 `AITrading` 背景模式
- 設定需記住上次值（包含：`K線週期 / K線顯示根數 / K線即時更新秒數 / 重訓最大樣本數 / 止盈止損 / 風險偏好`）
- 左側移除「環境自檢」按鈕
- 抓資料與訓練改為「全週期同步」：`5m / 15m / 30m / 1h / 1d` 各自抓資料、各自訓練、各自輸出模型
- 每個週期只保留最新 `5000` 根 K 線進行更新與訓練（避免資料量過大）
- 事件風控：每小時追蹤「美聯儲 / 川普 / 戰爭 / 恐慌」新聞關鍵字，事件特徵會寫入 `signals_with_features_*.csv` 供 AI 訓練
- 事件分頁改為「左：過去事件、右：未來事件」，未來事件包含 CPI/PPI/FOMC 預估時程（演算法推算）

快速更新效能：
- `快速更新` 只會在最近窗口（預設 60 天）上做特徵/推論/回測，用來達成「更新很快」的目標。
- 可用 `.env` 的 `QUICK_WINDOW_DAYS` 調整窗口大小（越小越快）。
- OHLCV 快取以 `data/*.parquet` 為主（效能較佳），同時仍會輸出 CSV 以相容舊工具。

## OKX 永續合約（SWAP）模擬盤交易（Paper）
你要的是永續合約交易，請使用 OKX 的 SWAP：
- `OKX_INST_TYPE=SWAP`
- `OKX_INST_ID=BTC-USDT-SWAP`

### 1) 必填金鑰
在 `.env` 設定以下三個必填：
- `OKX_API_KEY`
- `OKX_API_SECRET`
- `OKX_API_PASSPHRASE`

### 2) 模擬盤（預設）
本專案用 OKX v5 模擬盤 header：`x-simulated-trading: 1`。

安全預設：不送單（dry-run，只輸出會送出的 payload）
```bash
python okx_paper_trade.py --symbol BTCUSDT --interval 1h
```

允許送出「模擬盤」下單（仍是 simulated）
```bash
set OKX_ENABLE_TRADING=1
python okx_paper_trade.py --symbol BTCUSDT --interval 1h
```

你也可以在 Streamlit 儀表板左側「OKX 模擬盤交易」直接觸發下單並查看回應。

### 3) 槓桿與倉位模式注意事項
- 槓桿上限：`OKX_MAX_LEVERAGE=100`
- 保證金模式：`OKX_TD_MODE=isolated`（或 `cross`）
- 倉位模式：`OKX_POS_MODE=net` 或 `long_short`
  - `net`：通常不需要 `posSide`
  - `long_short`：OKX 對 SWAP/FUTURES 下單常需要 `posSide=long/short`

### 4) 下單本金
用 `OKX_NOTIONAL_USDT` 控制每次下單本金（預設 50 USDT）。

## 輸出檔案
每個週期會分開輸出，避免互相覆蓋：
- `data/<SYMBOL>_<INTERVAL>_ohlcv.parquet`：OHLCV 快取
- `data/<SYMBOL>_<INTERVAL>_ohlcv.csv`：相容用快取
- `outputs/signals_with_features_<SYMBOL>_<INTERVAL>.csv`：特徵 + 模型訊號 + 槓桿
- `outputs/backtest_curve_<SYMBOL>_<INTERVAL>.csv`：權益/回撤曲線
- `outputs/report_<SYMBOL>_<INTERVAL>.json`：摘要指標 + 最新決策
- `models/<SYMBOL>_<INTERVAL>/*.joblib`：各週期模型

相容輸出（永遠是「最後一次跑的結果」）：
- `outputs/signals_with_features.csv`
- `outputs/backtest_curve.csv`
- `outputs/report.json`

## 桌面 UI（舊版）
啟動：
```bash
python app_gui.py
```

打包成單一 EXE（Windows）：
```powershell
powershell -ExecutionPolicy Bypass -File build_tools\build_exe.ps1
```

輸出：
- `dist/BTC_AI_Backtest.exe`

## 風險提示
這是研究/回測/自動化實驗用途。槓桿交易風險極高，請先用模擬盤、最小倉位、多次驗證再考慮任何真實交易。

---

## 最新版更新（2026-05-21）

### Dashboard 修正
- 修正 K 線即時更新造成畫面重複渲染、頁面無限延伸的問題。
- 自動刷新機制改為非阻塞前端定時 reload，避免 `sleep + rerun` 疊加 UI。
- 修正分頁標題偶發重複顯示問題。

### 圖表互動
- 統一主要圖表互動：`拖曳 = 平移 (pan)`、`滾輪 = 縮放 (zoom)`。
- 保留 modebar（Pan / Zoom / Reset）按鈕。

### 交易與顯示
- 交易紀錄累計圖改為僅顯示「台北時區今日 00:00 之後」資料。
- 信心指數改為方向信心（取 `max(看漲, 看跌)`），避免觀望偏高時長期顯示 0。
- CSV 讀取加入容錯，壞行會自動略過，避免整個 dashboard 因單行格式錯誤中斷。

## 專案狀態
- AI 蒸餾：處理中（進行中）

