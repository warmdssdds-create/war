@echo off
:: Run the event-driven backtest on synthetic sample data.
:: Pass a CSV file path as the first argument to use real candle data:
::   run_backtest.bat path\to\candles_15m.csv
pip install -r requirements.txt >nul 2>&1
python -m src.backtest %*
