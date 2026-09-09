// AI Trading Robot - Web Dashboard
// Handles API communication and UI updates

const API_BASE = '/api';
let statusInterval = null;

// ============================================================================
// API Functions
// ============================================================================

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.message || 'API Error');
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        addLog(`Error: ${error.message}`, 'error');
        throw error;
    }
}

// ============================================================================
// UI Update Functions
// ============================================================================

function addLog(message, level = 'info') {
    const logContainer = document.getElementById('logContainer');
    const timestamp = new Date().toLocaleTimeString();
    const className = `text-${level === 'error' ? 'danger' : level === 'warning' ? 'warning' : 'success'}`;
    
    const logEntry = document.createElement('div');
    logEntry.className = className;
    logEntry.textContent = `[${timestamp}] ${message}`;
    
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateStatus(status) {
    // Update main status
    document.getElementById('statusText').textContent = status.status || 'Unknown';
    document.getElementById('tradingText').textContent = status.is_running ? 'Active' : 'Inactive';
    document.getElementById('modeText').textContent = status.mode || '-';
    document.getElementById('symbolText').textContent = status.symbol || '-';
    document.getElementById('modelText').textContent = status.model_type || '-';
    
    // Update badge
    const badge = document.getElementById('statusBadge');
    if (status.is_running) {
        badge.className = 'badge bg-success';
        badge.textContent = 'Trading';
    } else if (status.status === 'initialized' || status.status === 'ready') {
        badge.className = 'badge bg-info';
        badge.textContent = 'Ready';
    } else {
        badge.className = 'badge bg-warning';
        badge.textContent = 'Idle';
    }
    
    // Update account status
    if (status.account) {
        document.getElementById('balanceText').textContent = `$${status.account.balance.toFixed(2)}`;
        document.getElementById('positionsText').textContent = status.account.open_positions;
        document.getElementById('tradesText').textContent = status.account.closed_trades;
    }
    
    // Update button states
    const isInitialized = status.status && status.status !== 'not_initialized';
    document.getElementById('trainBtn').disabled = !isInitialized;
    document.getElementById('backtestBtn').disabled = !isInitialized;
    document.getElementById('startBtn').disabled = !isInitialized || status.is_running;
    document.getElementById('stopBtn').disabled = !status.is_running;
}

async function refreshStatus() {
    try {
        const status = await apiCall('/status');
        updateStatus(status);
    } catch (error) {
        console.error('Failed to refresh status');
    }
}

// ============================================================================
// Button Handlers
// ============================================================================

document.getElementById('initBtn').addEventListener('click', async () => {
    try {
        const mode = document.getElementById('modeSelect').value;
        document.getElementById('initBtn').disabled = true;
        addLog(`Initializing bot in ${mode} mode...`);
        
        await apiCall('/init', 'POST', { mode });
        addLog('Bot initialized successfully!', 'success');
        
        await refreshStatus();
    } catch (error) {
        addLog('Failed to initialize bot', 'error');
    } finally {
        document.getElementById('initBtn').disabled = false;
    }
});

document.getElementById('trainBtn').addEventListener('click', async () => {
    try {
        const days = parseInt(document.getElementById('daysInput').value);
        document.getElementById('trainBtn').disabled = true;
        addLog(`Training model with ${days} days of historical data...`);
        
        await apiCall('/train', 'POST', { days_back: days });
        addLog('Model trained successfully!', 'success');
        
        await refreshStatus();
    } catch (error) {
        addLog('Failed to train model', 'error');
    } finally {
        document.getElementById('trainBtn').disabled = false;
    }
});

document.getElementById('backtestBtn').addEventListener('click', async () => {
    try {
        document.getElementById('backtestBtn').disabled = true;
        addLog('Running backtest...');
        
        const result = await apiCall('/backtest', 'POST', { days_back: 30 });
        addLog('Backtest completed!', 'success');
        
        await refreshStatus();
    } catch (error) {
        addLog('Backtest failed', 'error');
    } finally {
        document.getElementById('backtestBtn').disabled = false;
    }
});

document.getElementById('startBtn').addEventListener('click', async () => {
    try {
        document.getElementById('startBtn').disabled = true;
        addLog('Starting trading...');
        
        await apiCall('/start-trading', 'POST', { update_interval: 3600 });
        addLog('Trading started!', 'success');
        
        // Start auto-refresh
        if (statusInterval) clearInterval(statusInterval);
        statusInterval = setInterval(refreshStatus, 5000);
        
        await refreshStatus();
    } catch (error) {
        addLog('Failed to start trading', 'error');
        document.getElementById('startBtn').disabled = false;
    }
});

document.getElementById('stopBtn').addEventListener('click', async () => {
    try {
        document.getElementById('stopBtn').disabled = true;
        addLog('Stopping trading...');
        
        await apiCall('/stop-trading', 'POST');
        addLog('Trading stopped!', 'success');
        
        // Stop auto-refresh
        if (statusInterval) clearInterval(statusInterval);
        
        await refreshStatus();
    } catch (error) {
        addLog('Failed to stop trading', 'error');
    } finally {
        document.getElementById('stopBtn').disabled = false;
    }
});

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    addLog('Dashboard loaded');
    refreshStatus();
    
    // Refresh status every 10 seconds
    setInterval(refreshStatus, 10000);
});
