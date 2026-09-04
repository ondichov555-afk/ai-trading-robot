"""
Trading Engine Module - Automated entry and exit logic
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config


class TradeSignal:
    def __init__(self, signal_type, price, confidence, reason=""):
        self.signal_type = signal_type  # "BUY" or "SELL"
        self.price = price
        self.confidence = confidence  # 0-1
        self.reason = reason
        self.timestamp = datetime.now()


class Position:
    def __init__(self, position_id, signal, entry_price, quantity):
        self.position_id = position_id
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = None
        self.profit_loss_percent = None
        self.status = "OPEN"  # OPEN, CLOSED
        self.exit_reason = None
        
        # Risk management
        self.stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
        self.take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
        self.highest_price = entry_price
    
    def update_price(self, current_price):
        """Update position with current price"""
        if self.status == "OPEN":
            if current_price > self.highest_price:
                self.highest_price = current_price
                # Update trailing stop loss
                if config.TRAILING_STOP:
                    self.stop_loss = current_price * (1 - config.TRAILING_STOP_PERCENT / 100)
    
    def close(self, exit_price, reason=""):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.status = "CLOSED"
        self.exit_reason = reason
        
        # Calculate P&L
        self.profit_loss = (exit_price - self.entry_price) * self.quantity
        self.profit_loss_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        
        return self.profit_loss
    
    def get_info(self):
        """Get position information"""
        return {
            "position_id": self.position_id,
            "status": self.status,
            "entry_price": self.entry_price,
            "current_price": self.exit_price if self.status == "CLOSED" else None,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "profit_loss": self.profit_loss,
            "profit_loss_percent": self.profit_loss_percent
        }


class TradingEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.closed_positions = []
        self.trades_count = 0
        self.daily_loss = 0
    
    def generate_entry_signal(self, data, ai_prediction, technical_indicators):
        """
        Generate BUY/SELL signals based on multiple factors
        """
        try:
            if len(data) < 2:
                return None
            
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            current_macd = data['MACD'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            # AI prediction confidence
            ai_confidence = ai_prediction if ai_prediction else 0.5
            
            # BUY Signals
            buy_signals = 0
            buy_confidence = 0
            
            # Signal 1: AI model predicts price up
            if ai_confidence > config.ENTRY_THRESHOLD:
                buy_signals += 1
                buy_confidence += ai_confidence
            
            # Signal 2: RSI oversold (RSI < 30)
            if current_rsi < 30:
                buy_signals += 1
                buy_confidence += 0.7
            
            # Signal 3: Price above 20-day SMA
            if current_price > sma_20:
                buy_signals += 1
                buy_confidence += 0.6
            
            # Signal 4: MACD positive
            if current_macd > 0:
                buy_signals += 1
                buy_confidence += 0.6
            
            # SELL Signals
            sell_signals = 0
            sell_confidence = 0
            
            # Signal 1: AI model predicts price down
            if ai_confidence < (1 - config.ENTRY_THRESHOLD):
                sell_signals += 1
                sell_confidence += (1 - ai_confidence)
            
            # Signal 2: RSI overbought (RSI > 70)
            if current_rsi > 70:
                sell_signals += 1
                sell_confidence += 0.7
            
            # Signal 3: Price below 20-day SMA
            if current_price < sma_20:
                sell_signals += 1
                sell_confidence += 0.6
            
            # Signal 4: MACD negative
            if current_macd < 0:
                sell_signals += 1
                sell_confidence += 0.6
            
            # Determine final signal
            if buy_signals >= 3:
                final_confidence = min(buy_confidence / buy_signals, 1.0)
                return TradeSignal(
                    "BUY",
                    current_price,
                    final_confidence,
                    f"Buy signals: {buy_signals} | AI: {ai_confidence:.2f} | RSI: {current_rsi:.1f}"
                )
            
            elif sell_signals >= 3:
                final_confidence = min(sell_confidence / sell_signals, 1.0)
                return TradeSignal(
                    "SELL",
                    current_price,
                    final_confidence,
                    f"Sell signals: {sell_signals} | AI: {ai_confidence:.2f} | RSI: {current_rsi:.1f}"
                )
            
            return None
            
        except Exception as e:
            print(f"✗ Error generating signal: {str(e)}")
            return None
    
    def execute_trade(self, signal):
        """
        Execute trade based on signal
        """
        try:
            if signal is None or signal.confidence < config.ENTRY_THRESHOLD:
                return False
            
            # Check daily loss limit
            if abs(self.daily_loss) > (self.initial_balance * config.MAX_DAILY_LOSS_PERCENT / 100):
                print(f"⚠ Daily loss limit reached: {self.daily_loss:.2f}")
                return False
            
            # Check max open positions
            open_positions = [p for p in self.positions if p.status == "OPEN"]
            if len(open_positions) >= config.MAX_OPEN_POSITIONS:
                print(f"⚠ Max open positions reached: {len(open_positions)}")
                return False
            
            if signal.signal_type == "BUY":
                # Calculate position size
                position_size = self.balance * config.POSITION_SIZE
                quantity = position_size / signal.price
                
                # Create position
                position = Position(
                    len(self.positions),
                    signal,
                    signal.price,
                    quantity
                )
                
                self.positions.append(position)
                self.balance -= position_size
                self.trades_count += 1
                
                print(f"\n✓ BUY Order Executed")
                print(f"  Entry Price: ${signal.price:.2f}")
                print(f"  Quantity: {quantity:.4f}")
                print(f"  Stop Loss: ${position.stop_loss:.2f}")
                print(f"  Take Profit: ${position.take_profit:.2f}")
                print(f"  Reason: {signal.reason}")
                
                return True
            
            elif signal.signal_type == "SELL":
                # Close all open positions
                closed_count = 0
                for position in self.positions:
                    if position.status == "OPEN":
                        pnl = position.close(signal.price, f"Sell signal: {signal.reason}")
                        self.balance += (position.quantity * signal.price)
                        self.closed_positions.append(position)
                        self.daily_loss += pnl
                        closed_count += 1
                
                if closed_count > 0:
                    print(f"\n✓ SELL Order Executed")
                    print(f"  Exit Price: ${signal.price:.2f}")
                    print(f"  Positions Closed: {closed_count}")
                    print(f"  Reason: {signal.reason}")
                
                return closed_count > 0
            
        except Exception as e:
            print(f"✗ Error executing trade: {str(e)}")
            return False
    
    def check_exit_conditions(self, current_price):
        """
        Check and execute exit conditions (SL, TP)
        """
        for position in self.positions:
            if position.status == "OPEN":
                position.update_price(current_price)
                
                # Check Stop Loss
                if current_price <= position.stop_loss:
                    pnl = position.close(current_price, "Stop Loss Hit")
                    self.balance += (position.quantity * current_price)
                    self.closed_positions.append(position)
                    self.daily_loss += pnl
                    print(f"\n⛔ Stop Loss Hit - Position {position.position_id} closed")
                    print(f"  Exit Price: ${current_price:.2f}")
                    print(f"  P&L: ${pnl:.2f} ({position.profit_loss_percent:.2f}%)")
                
                # Check Take Profit
                elif current_price >= position.take_profit:
                    pnl = position.close(current_price, "Take Profit Hit")
                    self.balance += (position.quantity * current_price)
                    self.closed_positions.append(position)
                    self.daily_loss += pnl
                    print(f"\n✅ Take Profit Hit - Position {position.position_id} closed")
                    print(f"  Exit Price: ${current_price:.2f}")
                    print(f"  P&L: ${pnl:.2f} ({position.profit_loss_percent:.2f}%)")
    
    def get_account_status(self):
        """
        Get current account status
        """
        open_positions = [p for p in self.positions if p.status == "OPEN"]
        
        return {
            "balance": self.balance,
            "open_positions": len(open_positions),
            "closed_trades": len(self.closed_positions),
            "total_trades": self.trades_count,
            "daily_loss": self.daily_loss,
            "positions": [p.get_info() for p in open_positions]
        }


# Example usage
if __name__ == "__main__":
    engine = TradingEngine(initial_balance=10000)
    print(f"Trading Engine initialized with balance: ${engine.balance:.2f}")
