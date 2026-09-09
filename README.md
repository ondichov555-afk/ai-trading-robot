# AI Trading Robot

AI-powered trading robot with machine learning models, backtesting engine, and live trading capabilities.

## Features

✨ **AI-Powered Trading**
- Machine learning models for price prediction
- Multiple trading strategies
- Real-time market analysis
- Paper and live trading modes

📊 **Backtesting Engine**
- Historical performance testing
- Strategy optimization
- Risk analysis and metrics
- Equity curve visualization

🔧 **Technical Analysis**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Moving averages
- Volume analysis

🖥️ **Web Dashboard**
- Real-time monitoring
- One-click trading controls
- Live performance metrics
- Activity logging
- Account status tracking

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/ondichov555-afk/ai-trading-robot.git
cd ai-trading-robot

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### CLI Mode (Interactive)
```bash
python main.py
```

#### Web Dashboard
```bash
python app.py
```
Then open http://localhost:5000 in your browser

### Configuration

Edit `config.py` to customize:
- Trading symbol (default: BTC/USDT)
- Timeframe (default: 1h)
- Model type (LSTM or GRU)
- Risk parameters
- Live trading settings

## Project Structure

```
ai-trading-robot/
├── main.py              # Core trading bot
├── app.py               # Flask web application
├── ai_model.py          # Machine learning models
├── trading_engine.py    # Trade execution logic
├── data_fetcher.py      # Market data retrieval
├── backtester.py        # Backtesting engine
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── templates/           # HTML templates
│   └── index.html      # Dashboard UI
├── static/              # Frontend assets
│   ├── css/
│   │   └── style.css   # Dashboard styling
│   └── js/
│       └── app.js      # Dashboard logic
└── README.md           # Documentation
```

## API Endpoints

### Bot Control
- `POST /api/init` - Initialize bot
- `POST /api/train` - Train AI model
- `POST /api/backtest` - Run backtest
- `POST /api/start-trading` - Start trading
- `POST /api/stop-trading` - Stop trading

### Status & Info
- `GET /api/status` - Get bot status
- `GET /api/account` - Get account details
- `GET /health` - Health check

## Dashboard Features

- **Control Panel**: Initialize, train, and control trading
- **Status Display**: Real-time bot and market status
- **Account Metrics**: Balance, positions, and trades
- **Activity Log**: Timestamped event tracking
- **Auto-refresh**: Live data updates every 10 seconds

## Trading Modes

### Paper Trading (Simulation)
- Risk-free testing
- Full strategy validation
- Performance metrics tracking

### Live Trading (Real Money)
- Real market execution
- Actual profit/loss
- ⚠️ Use with caution - test thoroughly first!

## Backtesting

Test your strategy on historical data:
```python
bot = TradingBot(mode="paper")
results = bot.run_backtest(days_back=30)
```

## Risk Management

- Stop-loss orders
- Position sizing
- Maximum daily loss limits
- Portfolio rebalancing

## Technologies Used

- **ML Framework**: TensorFlow/Keras
- **Data Fetching**: yfinance, CCXT
- **Technical Analysis**: TA-Lib
- **Web Framework**: Flask
- **Frontend**: Bootstrap 5, Chart.js
- **Data Processing**: Pandas, NumPy

## Performance Metrics

- Sharpe Ratio
- Win Rate
- Drawdown Analysis
- Risk-Adjusted Returns

## Disclaimer

⚠️ **WARNING**: Cryptocurrency trading involves substantial risk of loss. Past performance is not indicative of future results. This bot is provided as-is for educational and research purposes. Use at your own risk and with thorough testing.

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open a GitHub issue.

---

**Last Updated**: September 2026
**Status**: Active Development
