"""
Main Trading Bot - Automated AI Trading Robot
Handles data fetching, model training, and live/paper trading
"""

import time
import sys
from datetime import datetime
import config
from data_fetcher import DataFetcher
from ai_model import AITradingModel
from trading_engine import TradingEngine
from backtester import Backtester


class TradingBot:
    def __init__(self, mode="paper"):
        """
        Initialize Trading Bot
        mode: "paper" for paper trading, "live" for live trading
        """
        self.mode = mode
        self.data_fetcher = DataFetcher(config.SYMBOL, config.TIMEFRAME)
        self.model = AITradingModel(config.MODEL_TYPE)
        self.engine = TradingEngine(initial_balance=10000)
        self.is_running = False
        
        print(f"\n{'='*60}")
        print(f"AI Trading Robot Initialized")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper()}")
        print(f"Symbol: {config.SYMBOL}")
        print(f"Timeframe: {config.TIMEFRAME}")
        print(f"Model: {config.MODEL_TYPE}")
        print(f"{'='*60}\n")
    
    def initialize_model(self, days_back=60):
        """
        Fetch historical data and train the AI model
        """
        print("\n[STEP 1] Fetching Historical Data...")
        print(f"Downloading {days_back} days of data...")
        
        # Fetch data
        data = self.data_fetcher.fetch_data(days_back=days_back)
        if data is None or len(data) == 0:
            print("✗ Failed to fetch data")
            return False
        
        # Add technical indicators
        print("\n[STEP 2] Adding Technical Indicators...")
        data = self.data_fetcher.add_technical_indicators(data)
        if data is None:
            print("✗ Failed to add indicators")
            return False
        
        # Prepare data for model training
        print("\n[STEP 3] Preparing Data for Model Training...")
        X, y = self.data_fetcher.get_data_for_ml(config.LOOKBACK_WINDOW)
        
        if X is None or len(X) == 0:
            print("✗ Failed to prepare training data")
            return False
        
        print(f"Training samples: {len(X)}")
        
        # Train model
        print("\n[STEP 4] Training AI Model...")
        print(f"Model Type: {config.MODEL_TYPE}")
        self.model.train(X, y, epochs=30, batch_size=32)
        
        # Save model
        self.model.save_model("models/trading_model.h5")
        
        print("\n✓ Model initialization complete!")
        return True
    
    def run_backtest(self, days_back=30):
        """
        Run backtest on historical data
        """
        print("\n[BACKTEST] Running Strategy Backtest...")
        
        # Fetch data
        data = self.data_fetcher.fetch_data(days_back=days_back)
        if data is None:
            print("✗ Failed to fetch backtest data")
            return None
        
        data = self.data_fetcher.add_technical_indicators(data)
        if data is None:
            print("✗ Failed to add indicators")
            return None
        
        # Run backtest
        backtester = Backtester(initial_balance=10000)
        results = backtester.run_backtest(data, self.model)
        
        # Plot results
        print("\nGenerating backtest visualizations...")
        backtester.plot_equity_curve()
        
        return results
    
    def live_trading_loop(self, update_interval=3600):
        """
        Main live trading loop
        update_interval: seconds between updates (default 1 hour)
        """
        self.is_running = True
        print(f"\n{'='*60}")
        print(f"Starting Live Trading Loop")
        print(f"Update Interval: {update_interval} seconds")
        print(f"{'='*60}\n")
        
        try:
            iteration = 0
            while self.is_running:
                iteration += 1
                print(f"\n[ITERATION {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'-'*60}")
                
                # Fetch latest data
                print("Fetching latest market data...")
                data = self.data_fetcher.fetch_data(days_back=30)
                
                if data is None or len(data) < config.LOOKBACK_WINDOW:
                    print("✗ Insufficient data, waiting for next update...")
                    time.sleep(update_interval)
                    continue
                
                # Add indicators
                data = self.data_fetcher.add_technical_indicators(data)
                current_price = data['close'].iloc[-1]
                print(f"Current Price: ${current_price:.2f}")
                
                # Get AI prediction
                if self.model.is_trained:
                    X = data[['close', 'RSI', 'MACD', 'volume']].iloc[-config.LOOKBACK_WINDOW:].values
                    X = X.reshape(1, X.shape[0], X.shape[1])
                    ai_prediction = self.model.predict(X)
                    print(f"AI Prediction: {ai_prediction:.2f}")
                else:
                    ai_prediction = 0.5
                
                # Generate trading signal
                signal = self.engine.generate_entry_signal(data, ai_prediction, {})
                
                # Execute trade if signal is strong
                if signal and signal.confidence > config.ENTRY_THRESHOLD:
                    print(f"\n[SIGNAL] {signal.signal_type} Signal Generated")
                    print(f"Confidence: {signal.confidence:.2f}")
                    print(f"Reason: {signal.reason}")
                    
                    if config.USE_LIVE_TRADING:
                        self.engine.execute_trade(signal)
                    else:
                        print("(Paper Trading - No actual trade executed)")
                
                # Check exit conditions
                self.engine.check_exit_conditions(current_price)
                
                # Print account status
                status = self.engine.get_account_status()
                print(f"\n[ACCOUNT STATUS]")
                print(f"Balance: ${status['balance']:.2f}")
                print(f"Open Positions: {status['open_positions']}")
                print(f"Closed Trades: {status['closed_trades']}")
                
                print(f"\nNext update in {update_interval} seconds...")
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Trading bot stopped by user")
            self.is_running = False
        
        except Exception as e:
            print(f"\n✗ Error in trading loop: {str(e)}")
            self.is_running = False
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        print("Trading bot stopped")
    
    def print_menu(self):
        """Print main menu"""
        print(f"\n{'='*60}")
        print(f"AI Trading Robot - Main Menu")
        print(f"{'='*60}")
        print("1. Initialize & Train Model")
        print("2. Run Backtest")
        print("3. Start Paper Trading")
        print("4. View Account Status")
        print("5. Exit")
        print(f"{'='*60}")
    
    def run_interactive(self):
        """Interactive mode"""
        while True:
            self.print_menu()
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                days = input("Days of historical data (default 60): ").strip() or "60"
                self.initialize_model(days_back=int(days))
            
            elif choice == "2":
                print("\nRunning Backtest...")
                self.run_backtest(days_back=30)
            
            elif choice == "3":
                print("\nStarting Paper Trading...")
                print("Press Ctrl+C to stop")
                self.live_trading_loop(update_interval=60)  # Update every minute for demo
            
            elif choice == "4":
                status = self.engine.get_account_status()
                print(f"\n{'='*60}")
                print(f"Account Status")
                print(f"{'='*60}")
                print(f"Balance: ${status['balance']:.2f}")
                print(f"Open Positions: {status['open_positions']}")
                print(f"Closed Trades: {status['closed_trades']}")
                print(f"Daily Loss: ${status['daily_loss']:.2f}")
            
            elif choice == "5":
                print("Exiting...")
                break
            
            else:
                print("Invalid option")


def main():
    """Main entry point"""
    # Create trading bot
    bot = TradingBot(mode="paper")  # Use "live" for actual trading
    
    # Run interactive mode
    bot.run_interactive()


if __name__ == "__main__":
    main()
