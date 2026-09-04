"""
Backtesting Module - Test trading strategies on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import config
from data_fetcher import DataFetcher
from ai_model import AITradingModel
from trading_engine import TradingEngine


class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.engine = TradingEngine(initial_balance)
        self.backtest_results = []
        self.equity_curve = [initial_balance]
    
    def run_backtest(self, data, ai_model, verbose=True):
        """
        Run backtest on historical data
        """
        try:
            if data is None or len(data) < config.LOOKBACK_WINDOW:
                print("✗ Insufficient data for backtesting")
                return None
            
            print(f"\n{'='*60}")
            print(f"Starting Backtest")
            print(f"{'='*60}")
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Data Points: {len(data)}")
            print(f"Period: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
            print(f"{'='*60}\n")
            
            # Backtest loop
            for i in range(config.LOOKBACK_WINDOW, len(data)):
                current_data = data.iloc[:i+1].copy()
                current_price = data['close'].iloc[i]
                
                # Prepare data for AI prediction
                if i >= config.LOOKBACK_WINDOW:
                    X = current_data[['close', 'RSI', 'MACD', 'volume']].iloc[-config.LOOKBACK_WINDOW:].values
                    X = X.reshape(1, X.shape[0], X.shape[1])
                    
                    # Get AI prediction
                    ai_prediction = ai_model.predict(X) if ai_model.is_trained else 0.5
                else:
                    ai_prediction = 0.5
                
                # Generate entry signal
                signal = self.engine.generate_entry_signal(
                    current_data, 
                    ai_prediction,
                    {}
                )
                
                # Execute trade
                if signal and signal.confidence > config.ENTRY_THRESHOLD:
                    self.engine.execute_trade(signal)
                
                # Check exit conditions
                self.engine.check_exit_conditions(current_price)
                
                # Update equity curve
                self.equity_curve.append(self.engine.balance)
            
            # Print results
            return self.get_backtest_summary()
            
        except Exception as e:
            print(f"✗ Error during backtest: {str(e)}")
            return None
    
    def get_backtest_summary(self):
        """
        Get backtest summary statistics
        """
        try:
            total_return = self.engine.balance - self.initial_balance
            return_percent = (total_return / self.initial_balance) * 100
            
            closed_trades = self.engine.closed_positions
            winning_trades = [t for t in closed_trades if t.profit_loss > 0]
            losing_trades = [t for t in closed_trades if t.profit_loss < 0]
            
            win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
            
            avg_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
            
            # Max drawdown
            equity_array = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            # Profit factor
            gross_profit = sum([t.profit_loss for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t.profit_loss for t in losing_trades])) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            summary = {
                "final_balance": self.engine.balance,
                "total_return": total_return,
                "return_percent": return_percent,
                "total_trades": len(closed_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "max_consecutive_losses": self._max_consecutive_losses(closed_trades),
                "sharpe_ratio": self._calculate_sharpe_ratio()
            }
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Backtest Results")
            print(f"{'='*60}")
            print(f"Final Balance: ${summary['final_balance']:.2f}")
            print(f"Total Return: ${summary['total_return']:.2f} ({summary['return_percent']:.2f}%)")
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Winning Trades: {summary['winning_trades']}")
            print(f"Losing Trades: {summary['losing_trades']}")
            print(f"Win Rate: {summary['win_rate']:.2f}%")
            print(f"Avg Win: ${summary['avg_win']:.2f}")
            print(f"Avg Loss: ${summary['avg_loss']:.2f}")
            print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
            print(f"Profit Factor: {summary['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            print(f"{'='*60}\n")
            
            return summary
            
        except Exception as e:
            print(f"✗ Error calculating summary: {str(e)}")
            return None
    
    def _max_consecutive_losses(self, trades):
        """Calculate max consecutive losing trades"""
        if not trades:
            return 0
        
        max_loss_streak = 0
        current_loss_streak = 0
        
        for trade in trades:
            if trade.profit_loss < 0:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0
        
        return max_loss_streak
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe Ratio"""
        try:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            excess_returns = returns - (risk_free_rate / 252)
            
            if len(excess_returns) == 0:
                return 0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe
        except:
            return 0
    
    def plot_equity_curve(self, save_path="backtest_results/equity_curve.png"):
        """Plot equity curve"""
        try:
            plt.figure(figsize=(14, 6))
            plt.plot(self.equity_curve, linewidth=2, label='Equity')
            plt.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
            plt.xlabel('Trading Days')
            plt.ylabel('Account Balance ($)')
            plt.title('Equity Curve - Backtest Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"✓ Equity curve saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"✗ Error plotting equity curve: {str(e)}")
    
    def plot_trades(self, data, save_path="backtest_results/trades.png"):
        """Plot trades on price chart"""
        try:
            plt.figure(figsize=(14, 6))
            plt.plot(data['close'], label='Price', linewidth=2)
            
            # Plot entry points
            for position in self.engine.closed_positions:
                plt.scatter(data[data['timestamp'] == position.entry_time.strftime('%Y-%m-%d %H:%M:%S')].index,
                           position.entry_price, color='green', marker='^', s=100, label='Entry')
                plt.scatter(data[data['timestamp'] == position.exit_time.strftime('%Y-%m-%d %H:%M:%S')].index,
                           position.exit_price, color='red', marker='v', s=100, label='Exit')
            
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.title('Trading Signals on Price Chart')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"✓ Trade chart saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"✗ Error plotting trades: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Fetch data
    fetcher = DataFetcher("BTC-USD", "1h")
    data = fetcher.fetch_data(days_back=30)
    data = fetcher.add_technical_indicators()
    
    # Train model
    model = AITradingModel("LSTM")
    X, y = fetcher.get_data_for_ml()
    model.train(X, y, epochs=20)
    
    # Run backtest
    backtester = Backtester(initial_balance=10000)
    results = backtester.run_backtest(data, model)
    
    # Plot results
    backtester.plot_equity_curve()
    backtester.plot_trades(data)
