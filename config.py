"""
Configuration settings for AI Trading Robot
"""

# Market Data Settings
SYMBOL = "BTC/USDT"  # Trading pair
TIMEFRAME = "1h"  # 1h, 4h, 1d
LOOKBACK_PERIOD = 100  # Number of candles for analysis

# Entry/Exit Settings
ENTRY_THRESHOLD = 0.65  # AI confidence for entry (0-1)
EXIT_THRESHOLD = 0.45  # AI confidence for exit (0-1)
TAKE_PROFIT_PERCENT = 3.0  # 3% profit target
STOP_LOSS_PERCENT = 1.5  # 1.5% stop loss
TRAILING_STOP = True
TRAILING_STOP_PERCENT = 0.5

# Risk Management
POSITION_SIZE = 0.1  # 10% of portfolio per trade
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS_PERCENT = 5.0

# Model Settings
MODEL_TYPE = "LSTM"  # LSTM, GRU, or RandomForest
LOOKBACK_WINDOW = 20  # Number of previous candles for prediction
TRAIN_TEST_SPLIT = 0.8

# API Settings
USE_LIVE_TRADING = False  # Set to True for live trading
EXCHANGE = "binance"  # Exchange to use
API_KEY = ""  # Add your API key
API_SECRET = ""  # Add your API secret

# Notification Settings
SEND_ALERTS = True
ALERT_EMAIL = ""
SLACK_WEBHOOK = ""
