"""
Flask Web Application for AI Trading Robot
Provides REST API and web dashboard for monitoring and controlling the trading bot
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import logging
from datetime import datetime
from threading import Thread
import sys
import config
from main import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global trading bot instance
trading_bot = None
trading_thread = None


class BotManager:
    """Manages trading bot lifecycle"""
    def __init__(self):
        self.bot = None
        self.is_running = False
        self.status = "idle"
        self.messages = []
    
    def initialize(self, mode="paper"):
        """Initialize the trading bot"""
        try:
            self.bot = TradingBot(mode=mode)
            self.status = "initialized"
            self.log(f"Bot initialized in {mode} mode")
            return True
        except Exception as e:
            self.log(f"Error initializing bot: {str(e)}", level="error")
            return False
    
    def train_model(self, days_back=60):
        """Train the AI model"""
        if not self.bot:
            self.log("Bot not initialized", level="error")
            return False
        
        try:
            self.status = "training"
            self.log(f"Starting model training with {days_back} days of data...")
            result = self.bot.initialize_model(days_back=days_back)
            self.status = "ready" if result else "error"
            return result
        except Exception as e:
            self.log(f"Error training model: {str(e)}", level="error")
            self.status = "error"
            return False
    
    def start_trading(self, update_interval=3600):
        """Start the trading loop in a background thread"""
        if not self.bot:
            self.log("Bot not initialized", level="error")
            return False
        
        if self.is_running:
            self.log("Trading already running", level="warning")
            return False
        
        def run_bot():
            try:
                self.is_running = True
                self.status = "trading"
                self.log("Started live trading loop")
                self.bot.live_trading_loop(update_interval=update_interval)
            except Exception as e:
                self.log(f"Error in trading loop: {str(e)}", level="error")
                self.status = "error"
            finally:
                self.is_running = False
        
        thread = Thread(target=run_bot, daemon=True)
        thread.start()
        return True
    
    def stop_trading(self):
        """Stop the trading loop"""
        if not self.is_running:
            self.log("Trading not running", level="warning")
            return False
        
        try:
            self.bot.stop()
            self.is_running = False
            self.status = "stopped"
            self.log("Stopped trading loop")
            return True
        except Exception as e:
            self.log(f"Error stopping bot: {str(e)}", level="error")
            return False
    
    def get_status(self):
        """Get current bot status"""
        if not self.bot:
            return {
                "status": "not_initialized",
                "is_running": False,
                "account": None
            }
        
        account = self.bot.engine.get_account_status()
        return {
            "status": self.status,
            "is_running": self.is_running,
            "mode": self.bot.mode,
            "symbol": config.SYMBOL,
            "timeframe": config.TIMEFRAME,
            "model_type": config.MODEL_TYPE,
            "account": account,
            "messages": self.messages[-50:]  # Last 50 messages
        }
    
    def log(self, message, level="info"):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.messages.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        logger.log(getattr(logging, level.upper(), logging.INFO), message)


# Initialize bot manager
manager = BotManager()


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize the trading bot"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'paper')
        
        if manager.initialize(mode=mode):
            return jsonify({"success": True, "message": f"Bot initialized in {mode} mode"})
        else:
            return jsonify({"success": False, "message": "Failed to initialize bot"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    """Train the AI model"""
    try:
        data = request.get_json() or {}
        days_back = data.get('days_back', 60)
        
        if manager.train_model(days_back=days_back):
            return jsonify({"success": True, "message": "Model training completed"})
        else:
            return jsonify({"success": False, "message": "Model training failed"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run backtest"""
    try:
        if not manager.bot:
            return jsonify({"success": False, "message": "Bot not initialized"}), 400
        
        data = request.get_json() or {}
        days_back = data.get('days_back', 30)
        
        results = manager.bot.run_backtest(days_back=days_back)
        return jsonify({
            "success": True,
            "message": "Backtest completed",
            "results": results if results else {}
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/start-trading', methods=['POST'])
def api_start_trading():
    """Start trading"""
    try:
        data = request.get_json() or {}
        update_interval = data.get('update_interval', 3600)
        
        if manager.start_trading(update_interval=update_interval):
            return jsonify({"success": True, "message": "Trading started"})
        else:
            return jsonify({"success": False, "message": "Failed to start trading"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/stop-trading', methods=['POST'])
def api_stop_trading():
    """Stop trading"""
    try:
        if manager.stop_trading():
            return jsonify({"success": True, "message": "Trading stopped"})
        else:
            return jsonify({"success": False, "message": "Trading not running"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get bot status"""
    return jsonify(manager.get_status())


@app.route('/api/account', methods=['GET'])
def api_account():
    """Get account details"""
    if not manager.bot:
        return jsonify({"success": False, "message": "Bot not initialized"}), 400
    
    account = manager.bot.engine.get_account_status()
    return jsonify({
        "success": True,
        "account": account
    })


# ============================================================================
# WEB DASHBOARD ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Trading dashboard"""
    return render_template('index.html')


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "message": "Internal server error"}), 500


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI Trading Robot - Web Application")
    print("="*60)
    print("Starting Flask server...")
    print("Dashboard: http://localhost:5000")
    print("API Docs: http://localhost:5000/api")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
