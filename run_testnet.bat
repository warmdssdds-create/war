@echo off
:: Start the trading bot in testnet mode.
:: Requires a valid .env file with Bybit testnet API credentials.
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example -- fill in your testnet API credentials before restarting.
    exit /b 1
)
pip install -r requirements.txt >nul 2>&1
python -m src.main --env testnet
