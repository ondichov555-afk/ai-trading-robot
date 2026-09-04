"""
Data Fetcher Module - Retrieves market data for trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical Analysis library
import config


class DataFetcher:
    def __init__(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
    
    def fetch_data(self, days_back=30):
        """
        Fetch historical market data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # For crypto, use yfinance
            ticker = self.symbol.replace("/", "-")
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                interval='1h' if self.timeframe == '1h' else '1d'
            )
            
            if data.empty:
                print(f"Warning: No data fetched for {self.symbol}")
                return None
            
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            self.data = data
            print(f"✓ Fetched {len(data)} candles for {self.symbol}")
            return data
            
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            return None
    
    def add_technical_indicators(self, data=None):
        """
        Add technical indicators to the data
        """
        if data is None:
            data = self.data.copy()
        
        if data is None or len(data) == 0:
            return None
        
        try:
            # Moving Averages
            data['SMA_20'] = ta.trend.sma_indicator(close=data['close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(close=data['close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(close=data['close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(close=data['close'], window=26)
            
            # Momentum Indicators
            data['RSI'] = ta.momentum.rsi(close=data['close'], window=14)
            data['MACD'] = ta.trend.macd_diff(close=data['close'])
            
            # Volatility
            data['ATR'] = ta.volatility.average_true_range(
                high=data['high'], 
                low=data['low'], 
                close=data['close'], 
                window=14
            )
            data['Bollinger_High'] = ta.volatility.bollinger_hband(
                close=data['close'], 
                window=20
            )
            data['Bollinger_Low'] = ta.volatility.bollinger_lband(
                close=data['close'], 
                window=20
            )
            
            # Volume
            data['Volume_SMA'] = ta.trend.sma_indicator(close=data['volume'], window=20)
            
            # Fill NaN values
            data = data.fillna(method='bfill').fillna(method='ffill')
            
            self.data = data
            print(f"✓ Added {len(data.columns) - 6} technical indicators")
            return data
            
        except Exception as e:
            print(f"✗ Error adding indicators: {str(e)}")
            return None
    
    def get_latest_data(self, n=1):
        """
        Get the latest n candles
        """
        if self.data is None or len(self.data) == 0:
            return None
        return self.data.tail(n)
    
    def get_data_for_ml(self, lookback_window=config.LOOKBACK_WINDOW):
        """
        Prepare data for machine learning model
        """
        if self.data is None or len(self.data) < lookback_window:
            return None
        
        # Create sequences for LSTM
        X = []
        y = []
        
        for i in range(len(self.data) - lookback_window):
            # Features: Close, RSI, MACD, SMA ratios
            features = self.data[['close', 'RSI', 'MACD', 'volume']].iloc[i:i+lookback_window].values
            X.append(features)
            
            # Target: 1 if price goes up, 0 if down
            future_price = self.data['close'].iloc[i+lookback_window]
            current_price = self.data['close'].iloc[i+lookback_window-1]
            y.append(1 if future_price > current_price else 0)
        
        return np.array(X), np.array(y)


# Example usage
if __name__ == "__main__":
    fetcher = DataFetcher("BTC-USD", "1h")
    
    # Fetch data
    data = fetcher.fetch_data(days_back=30)
    
    # Add indicators
    data_with_indicators = fetcher.add_technical_indicators()
    
    # Display latest data
    print("\nLatest Data:")
    print(fetcher.get_latest_data(5))
