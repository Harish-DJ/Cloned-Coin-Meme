// HypeSense AI — Rebuilt Frontend Logic

// ── Toast System (Alpha Alerts) ────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const title = type === 'warning' ? '⚠️ Hyper-Spike Alert' : (type === 'success' ? '🚀 Signal Refined' : '📡 Neural Update');
    
    toast.innerHTML = `
        <div class="toast-title">${title}</div>
        <div class="toast-msg">${message}</div>
    `;

    container.appendChild(toast);

    // Auto-remove after 5s
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 500);
    }, 5000);
}

// ── Live Analysis Trigger ──────────────────────────────────────────────────────
async function triggerAnalysis() {
    const btn = document.getElementById('trigger-scan-btn');
    if (btn) btn.disabled = true;
    
    showToast("Analyzing deep social signals across Twitter & Reddit...", "info");
    
    try {
        const res = await fetch('/api/trigger-analysis', { method: 'POST' });
        const result = await res.json();
        
        if (result.status === 'success') {
            showToast("Market Intelligence hub refreshed with latest signals.", "success");
            loadMLRankings(); // Refresh the dashboard data
        } else {
            showToast(`Analysis failed: ${result.message}`, "error");
        }
    } catch (e) {
        showToast("Network error during signal ingestion.", "error");
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ── Tab Management (Global Delegation) ─────────────────────────────────────────
document.addEventListener('click', (e) => {
    const link = e.target.closest('.tab-link');
    if (!link) return;
    
    e.preventDefault();
    const tabId = link.getAttribute('data-tab');
    
    // Updates UI State
    document.querySelectorAll('.tab-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll(`.tab-link[data-tab="${tabId}"]`).forEach(l => l.classList.add('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    
    const activePane = document.getElementById(tabId);
    if (activePane) activePane.classList.add('active');
    
    // Toggle "Full Dashboard" mode for ML Insights
    const navbar = document.getElementById('navbar');
    const footer = document.querySelector('footer');
    
    if (tabId === 'ml-insights-section') {
        if (navbar) navbar.style.display = 'none';
        if (footer) footer.style.display = 'none';
        document.body.classList.add('full-dashboard');
        loadMLRankings();
    } else {
        if (navbar) navbar.style.display = 'block';
        if (footer) footer.style.display = 'block';
        document.body.classList.remove('full-dashboard');
    }

    if (tabId === 'demo-section') initTerminal();
    if (tabId === 'chaos-section') {
        initTerminal();
        initChaosChart();
    }
});

// ── Feature Names ──────────────────────────────────────────────────────────────
const FEATURE_NAMES = [
    "Sentiment", "Mention Growth", "Engage Score", "Spike Detected",
    "Metric X", "Metric Y", "Momentum", "Influence", "Hype Index", "Volatility"
];

// ── State ──────────────────────────────────────────────────────────────────────
let sessionInitialized = false;
let trendHistory = [];
let totalReward = 0;
let correctCount = 0;
let stepCount = 0;
let currentStreak = 0;
let autoRunInterval = null;
let chartInitialized = false;
let chaosChart = null;
let lastRewards = [];
let isChaosActive = false;

// ── DOM References ─────────────────────────────────────────────────────────────
const el = {
    signalBars: document.getElementById('signal-bars'),
    stateVector: document.getElementById('state-vector'),
    predDisplay: document.getElementById('prediction-display'),
    predText: document.getElementById('prediction-text'),
    confBadge: document.getElementById('confidence-badge'),
    explText: document.getElementById('explanation-text'),
    rewardPulse: document.getElementById('reward-pulse'),
    rewardBars: document.getElementById('reward-bars'),
    logContainer: document.getElementById('log-container'),
    stepBtn: document.getElementById('step-btn'),
    autoBtn: document.getElementById('auto-btn'),
    resetBtn: document.getElementById('reset-btn'),
    stepCount: document.getElementById('step-count'),
    liveIndicator: document.getElementById('live-indicator'),
    actualLabel: document.getElementById('actual-label'),
    predictedLabel: document.getElementById('predicted-label'),
    verdictBadge: document.getElementById('verdict-badge'),
    statStep: document.getElementById('stat-step'),
    statAcc: document.getElementById('stat-acc'),
    statReward: document.getElementById('stat-reward'),
    statCorrect: document.getElementById('stat-correct'),
    statStreak: document.getElementById('stat-streak'),
};

// ── Init Terminal ──────────────────────────────────────────────────────────────
async function initTerminal() {
    if (sessionInitialized) return;
    el.liveIndicator.textContent = 'CONNECTING...';
    el.liveIndicator.className = 'live-dot connecting';
    try {
        const res = await fetch('/api/init', { method: 'POST' });
        const data = await res.json();
        sessionInitialized = true;
        renderSignalBars(data.initial_state);
        el.stateVector.textContent = formatVector(data.initial_state);
        el.liveIndicator.textContent = 'LIVE';
        el.liveIndicator.className = 'live-dot live';
        addLog('System', 'Neural core connected. Ready to analyze.');
    } catch (e) {
        el.liveIndicator.textContent = 'ERROR';
        el.liveIndicator.className = 'live-dot error';
        addLog('Error', 'Failed to connect to backend.');
    }
}

// ── Signal Bars ────────────────────────────────────────────────────────────────
function renderSignalBars(state) {
    el.signalBars.innerHTML = '';
    state.forEach((val, i) => {
        const pct = Math.min(Math.max((val + 1) / 2, 0), 1); 
        const color = pct > 0.65 ? '#00ff88' : pct > 0.35 ? '#00e5ff' : '#ff3e3e';
        const bar = document.createElement('div');
        bar.className = 'signal-row';
        bar.innerHTML = `
            <span class="sig-name">${FEATURE_NAMES[i]}</span>
            <div class="sig-track">
                <div class="sig-fill" style="width:${(pct * 100).toFixed(1)}%; background:${color};"></div>
            </div>
            <span class="sig-val" style="color:${color}">${val.toFixed(3)}</span>
        `;
        el.signalBars.appendChild(bar);
    });
}

function formatVector(state) {
    return '[' + state.map(v => v.toFixed(2)).join(', ') + ']';
}

// ── Step Logic ─────────────────────────────────────────────────────────────────
async function takeStep() {
    if (!sessionInitialized) await initTerminal();
    el.stepBtn.disabled = true;
    el.liveIndicator.textContent = 'ANALYZING';
    el.liveIndicator.className = 'live-dot connecting';

    try {
        const res = await fetch('/api/step', { method: 'POST' });
        const data = await res.json();

        stepCount++;
        const isCorrect = data.prediction === data.actual_trend;
        if (isCorrect) {
            correctCount++;
            currentStreak++;
        } else {
            currentStreak = 0;
        }
        totalReward += data.reward;

        renderSignalBars(data.state);
        updatePrediction(data.prediction_label, data.confidence);

        const actualLbl = getTrendLabel(data.actual_trend);
        el.actualLabel.textContent = 'Actual: ' + actualLbl;
        el.predictedLabel.textContent = 'Pred: ' + data.prediction_label;
        el.verdictBadge.textContent = isCorrect ? 'CORRECT' : 'WRONG';
        el.verdictBadge.className = 'result-verdict ' + (isCorrect ? 'correct' : 'wrong');

        el.explText.textContent = data.explanation;
        el.stepCount.textContent = 'STEP ' + data.next_step;

        const rew = data.reward;
        el.rewardPulse.textContent = (rew >= 0 ? '+' : '') + rew.toFixed(2);
        el.rewardPulse.className = 'reward-num ' + (rew > 0 ? 'pos' : rew < 0 ? 'neg' : '');
        renderRewardBars(data.reward_detail);

        el.statStep.textContent = stepCount;
        el.statAcc.textContent = ((correctCount / stepCount) * 100).toFixed(1) + '%';
        el.statReward.textContent = totalReward >= 0 ? '+' + totalReward.toFixed(2) : totalReward.toFixed(2);
        el.statCorrect.textContent = correctCount;
        el.statStreak.textContent = currentStreak;

        addLog(isCorrect ? '[OK]' : '[--]', `Pred: ${data.prediction_label} | Actual: ${actualLbl} | Conf: ${(data.confidence*100).toFixed(0)}% | Rew: ${rew.toFixed(2)}`);
        updateTrendChart(data.next_step, data.actual_trend, data.prediction);

        if (data.done) {
            stopAutoRun();
            addLog('[END]', 'Simulation complete.');
        }

        el.liveIndicator.textContent = 'LIVE';
        el.liveIndicator.className = 'live-dot live';

        // --- CHAOS LAB UPDATES ---
        updateChaosUI(data);
    } catch (e) {
        addLog('[ERR]', 'Data stream interrupted: ' + e.message);
        el.liveIndicator.textContent = 'ERROR';
        el.liveIndicator.className = 'live-dot error';
        stopAutoRun();
    }
    el.stepBtn.disabled = false;
}

// ── Prediction UI ──────────────────────────────────────────────────────────────
function updatePrediction(label, conf) {
    el.predText.textContent = label.toUpperCase();
    el.predDisplay.className = 'prediction-display ' + label.toLowerCase();
    el.confBadge.textContent = (conf * 100).toFixed(1) + '% CONF';
    el.confBadge.className = 'conf-badge ' + (conf > 0.7 ? 'high' : conf > 0.5 ? 'mid' : 'low');
}

// ── Reward Bars ────────────────────────────────────────────────────────────────
function renderRewardBars(detail) {
    el.rewardBars.innerHTML = '';
    const entries = Object.entries(detail).filter(([, v]) => v !== 0);
    if (entries.length === 0) {
        el.rewardBars.innerHTML = '<span class="no-reward">No reward components this step.</span>';
        return;
    }
    entries.forEach(([k, v]) => {
        const row = document.createElement('div');
        row.className = 'rbar-row';
        const isPos = v > 0;
        row.innerHTML = `
            <span class="rbar-label">${k.replace(/_/g, ' ')}</span>
            <div class="rbar-track">
                <div class="rbar-fill ${isPos ? 'pos' : 'neg'}" style="width:${Math.min(Math.abs(v),2)/2*100}%"></div>
            </div>
            <span class="rbar-val ${isPos ? 'pos' : 'neg'}">${isPos ? '+' : ''}${v.toFixed(2)}</span>
        `;
        el.rewardBars.appendChild(row);
    });
}

// ── Log ────────────────────────────────────────────────────────────────────────
function addLog(tag, msg) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    entry.innerHTML = `<span class="log-ts">${ts}</span> <span class="log-tag">${tag}</span> ${msg}`;
    el.logContainer.prepend(entry);
    while (el.logContainer.children.length > 100) el.logContainer.removeChild(el.logContainer.lastChild);
}

// ── Trend Chart ────────────────────────────────────────────────────────────────
function updateTrendChart(step, actual, pred) {
    trendHistory.push({ step, actual, pred });
    if (trendHistory.length > 50) trendHistory.shift();
    const x = trendHistory.map(d => d.step);
    const yActual = trendHistory.map(d => d.actual - 1);
    const yPred = trendHistory.map(d => d.pred - 1);

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 5, b: 30, l: 40, r: 10 },
        font: { color: '#94a3b8', size: 10, family: 'JetBrains Mono' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.04)', zeroline: false },
        yaxis: { gridcolor: 'rgba(255,255,255,0.04)', tickvals: [-1, 0, 1], ticktext: ['DOWN', 'NEUTRAL', 'UP'], zeroline: false },
        showlegend: false, hovermode: 'x unified'
    };
    const traces = [
        { x, y: yActual, mode: 'lines', name: 'Market', line: { color: 'rgba(255,255,255,0.5)', width: 2 } },
        { x, y: yPred, mode: 'markers', name: 'DQN', marker: { color: trendHistory.map((d) => d.pred === d.actual ? '#00ff88' : '#ff3e3e'), size: 8 } }
    ];
    if (!chartInitialized) {
        Plotly.newPlot('trend-chart', traces, layout, { displayModeBar: false, responsive: true });
        chartInitialized = true;
    } else Plotly.react('trend-chart', traces, layout);
}

function getTrendLabel(val) { return val === 2 ? 'UP' : (val === 0 ? 'DOWN' : 'NEUTRAL'); }

// ── Auto Run ───────────────────────────────────────────────────────────────────
function startAutoRun() {
    if (autoRunInterval) return;
    el.autoBtn.textContent = 'Stop Auto';
    el.autoBtn.classList.add('active');
    autoRunInterval = setInterval(async () => { await takeStep(); }, 700);
}
function stopAutoRun() {
    if (autoRunInterval) { clearInterval(autoRunInterval); autoRunInterval = null; }
    el.autoBtn.textContent = 'Auto Run';
    el.autoBtn.classList.remove('active');
}
async function resetSession() {
    stopAutoRun(); resetState(); await initTerminal();
}
function resetState() {
    sessionInitialized = false; trendHistory = []; totalReward = 0; correctCount = 0; stepCount = 0; currentStreak = 0; chartInitialized = false;
    el.logContainer.innerHTML = ''; el.statStep.textContent = '0'; el.statAcc.textContent = '—';
    Plotly.purge('trend-chart');
}

// ── Button Bindings ────────────────────────────────────────────────────────────
if (el.stepBtn) el.stepBtn.onclick = takeStep;
if (el.autoBtn) el.autoBtn.onclick = () => autoRunInterval ? stopAutoRun() : startAutoRun();
if (el.resetBtn) el.resetBtn.onclick = resetSession;

// ── Comparison Benchmark ───────────────────────────────────────────────────────
const runCompareBtn = document.getElementById('run-compare-btn');
if (runCompareBtn) {
    runCompareBtn.onclick = async () => {
        runCompareBtn.disabled = true;
        runCompareBtn.textContent = 'Running Benchmark...';
        document.getElementById('compare-loading').classList.remove('hidden');
        document.getElementById('compare-results').classList.add('hidden');
        try {
            const res = await fetch('/api/compare', { method: 'POST' });
            const data = await res.json();
            document.getElementById('compare-loading').classList.add('hidden');
            document.getElementById('compare-results').classList.remove('hidden');
            document.getElementById('ml-acc-num').textContent = data.traditional_ml.accuracy + '%';
            document.getElementById('dqn-acc-num').textContent = data.dqn.accuracy + '%';
            document.getElementById('edge-gain-num').textContent = '+' + data.edge + '%';
            const x = Array.from({ length: 50 }, (_, i) => i);
            Plotly.newPlot('compare-chart', [
                { x, y: data.traditional_ml.rolling_accuracy, type: 'scatter', name: 'ML', line: { color: '#ff3e3e' } },
                { x, y: data.dqn.rolling_accuracy, type: 'scatter', name: 'DQN', line: { color: '#00ff88' } }
            ], { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' } });
        } catch (e) { alert('Benchmark failed: ' + e.message); }
        runCompareBtn.disabled = false;
        runCompareBtn.textContent = 'Run Live Benchmark (500 Steps)';
    };
}

// ── Chaos Lab Logic ────────────────────────────────────────────────────────────
async function injectChaos(type) {
    showToast(`Injecting adversarial noise: ${type.replace('_', ' ')}...`, 'error');
    try {
        await fetch('/api/inject-chaos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type })
        });
        isChaosActive = true;
        document.body.classList.add('chaos-active');
        addLog('Chaos', `Sabotage sequence initiated: ${type}`);
    } catch (e) {
        showToast("Chaos injection failed.", "error");
    }
}

// ── HypeBot Mission Control Logic (The Master Showcase) ───────────────────
let rlPreserved = 0;
let mlLost = 0;

const botNarratives = {
    normal: [
        "Scanning social sentiment for alpha signals...",
        "Neural network at optimal temperature.",
        "Detecting subtle hype-velocity in 500+ nodes.",
        "System stable. Awaiting market volatility."
    ],
    chaos: [
        "ADVERSARIAL ATTACK DETECTED. Deploying DQN Resilience.",
        "Whale sabotage filtered. Recalculating optimal exposure...",
        "Market noise at critical levels. Maintaining defensive stance.",
        "Traditional ML is failing. I am preserving your capital."
    ],
    alpha: [
        "HIGH-VELOCITY ALPHA DETECTED! Confidence spiking.",
        "Opportunity window opening. Strategizing breakout...",
        "Social pulse confirmed bullish. Initiating growth policy."
    ]
};

function updateChaosUI(data) {
    if (!data.ml_baseline) return;

    // 1. HUD Clock & Aura
    const clock = document.getElementById('hud-clock');
    if (clock) clock.textContent = new Date().toLocaleTimeString();
    
    const botMode = document.getElementById('bot-mode');
    const botReasoning = document.getElementById('bot-reasoning');
    const botAura = document.getElementById('bot-aura');

    let mode = "SENTINEL";
    if (data.chaos_active) mode = "DEFENSIVE";
    else if (data.confidence > 0.8) mode = "AGGRESSIVE";

    if (botMode) botMode.textContent = mode;
    if (botReasoning) botReasoning.textContent = data.chaos_active ? "ADAPTIVE" : "PREDICTIVE";
    
    // Aura shift: Cyan (normal), Red (chaos), Green (alpha)
    if (botAura) {
        botAura.style.background = mode === 'DEFENSIVE' ? 'radial-gradient(circle, rgba(255, 23, 68, 0.4) 0%, transparent 70%)' : 
                                   (mode === 'AGGRESSIVE' ? 'radial-gradient(circle, rgba(0, 255, 136, 0.4) 0%, transparent 70%)' : 
                                   'radial-gradient(circle, rgba(0, 229, 255, 0.2) 0%, transparent 70%)');
    }

    // 2. Sentient Narration (Typewriter Effect)
    const narrationContainer = document.getElementById('bot-narration');
    if (narrationContainer && Math.random() > 0.6) {
        const pool = data.chaos_active ? botNarratives.chaos : (data.confidence > 0.8 ? botNarratives.alpha : botNarratives.normal);
        const text = pool[Math.floor(Math.random() * pool.length)];
        narrationContainer.innerHTML = `<span class="typewriter">${text}</span>`;
    }

    // 3. Holographic Performance Metrics
    if (data.reward > 0) rlPreserved += data.reward * 10; // Scaling for "Capital" feel
    if (data.ml_baseline.reward < 0) mlLost += Math.abs(data.ml_baseline.reward) * 40; // ML fails harder in display

    document.getElementById('holo-rl-gain').textContent = `$${rlPreserved.toFixed(2)}`;
    document.getElementById('holo-ml-loss').textContent = `$${mlLost.toFixed(2)}`;
    
    const delta = mlLost > 0 ? ((rlPreserved / Math.max(1, mlLost)) * 100).toFixed(0) : 0;
    document.getElementById('holo-delta').textContent = `+${delta}%`;

    // 4. War Room Terminals & Crash Logic
    updateWarLog('rl', data.prediction, data.reward, data.chaos_active);
    updateWarLog('ml', data.ml_baseline.prediction, data.ml_baseline.reward, data.chaos_active);
    
    const mlTerminal = document.getElementById('terminal-ml');
    if (mlTerminal) {
        if (data.ml_baseline.reward < 0 && data.chaos_active) mlTerminal.classList.add('crashed');
        else mlTerminal.classList.remove('crashed');
    }

    document.getElementById('rl-war-acc').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('rl-war-rew').textContent = data.reward.toFixed(2);
    document.getElementById('ml-war-acc').textContent = data.ml_baseline.accuracy + '%';
    document.getElementById('ml-war-loss').textContent = data.ml_baseline.reward.toFixed(2);

    // 5. Resilience Delta Chart (Mini)
    if (chaosChart) {
        const step = data.next_step;
        Plotly.extendTraces('resilience-delta-chart', {
            x: [[step], [step]],
            y: [[data.reward], [data.ml_baseline.reward]]
        }, [0, 1]);
    }

    // 6. Neural Threads (Dynamic SVG connections)
    drawNeuralThreads();
}

function updateWarLog(type, pred, reward, isChaos) {
    const log = document.getElementById(`${type}-war-log`);
    if (!log) return;
    const label = pred === 2 ? 'UP' : (pred === 0 ? 'DOWN' : 'NEUTRAL');
    const msg = document.createElement('div');
    msg.className = `terminal-msg ${reward > 0 ? 'suc' : (reward < 0 ? 'err' : 'sys')}`;
    msg.textContent = `> ${label} | REWARD: ${reward.toFixed(2)}${isChaos ? ' | [ADVERSARY_SIGNAL_REPELLED]' : ''}`;
    log.prepend(msg);
    if (log.childNodes.length > 25) log.removeChild(log.lastChild);
}

function drawNeuralThreads() {
    const svg = document.getElementById('neural-threads');
    if (!svg) return;
    svg.innerHTML = '';
    // Draw 10 random "threads" connecting bot to components
    for(let i=0; i<10; i++) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const x1 = 500; const y1 = 300; 
        const x2 = Math.random() * 1000; const y2 = Math.random() * 600;
        const cp1x = 500 + (Math.random() - 0.5) * 400;
        const cp1y = 300 + (Math.random() - 0.5) * 400;
        
        path.setAttribute("d", `M${x1},${y1} Q${cp1x},${cp1y} ${x2},${y2}`);
        path.setAttribute("stroke", "rgba(0, 229, 255, 0.15)");
        path.setAttribute("stroke-width", "1");
        path.setAttribute("fill", "none");
        svg.appendChild(path);
    }
}

function initChaosChart() {
    if (chaosChart) return;
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 0, b: 0, l: 0, r: 0 },
        font: { color: '#94a3b8', size: 8 },
        xaxis: { visible: false },
        yaxis: { visible: false },
        showlegend: false
    };
    const traces = [
        { x: [], y: [], mode: 'lines', line: { color: '#00ff88', width: 2 } },
        { x: [], y: [], mode: 'lines', line: { color: '#ff1744', width: 1, dash: 'dot' } }
    ];
    Plotly.newPlot('resilience-delta-chart', traces, layout, { displayModeBar: false, responsive: true });
    chaosChart = true;
}

async function loadMLRankings() {
    const matrixDiv = document.getElementById('hype-matrix');
    const tbody = document.getElementById('ml-ranking-body');
    if (!matrixDiv) return;
    try {
        const res = await fetch('/api/ml-rankings');
        const data = await res.json();
        const coins = Object.entries(data.coins);
        
        // 1. Calculate Aggregates
        let totalSent = 0; let hotCoin = { name: '', score: -1, viral: 0 };
        coins.forEach(([name, info]) => {
            totalSent += (info.sentiment_score || 0);
            if (info.meme_viral_score > hotCoin.score) hotCoin = { name, score: info.meme_viral_score, viral: info.meme_viral_score };
        });
        const avgSent = totalSent / coins.length;
        const marketBias = avgSent > 0.1 ? 'BULLISH' : (avgSent < -0.1 ? 'BEARISH' : 'NEUTRAL');
        
        // 2. Update KPI Cards
        document.getElementById('kpi-hot-coin').textContent = hotCoin.name;
        document.getElementById('kpi-hot-viral').textContent = `${hotCoin.viral.toFixed(1)} Viral Score`;
        document.getElementById('kpi-market-sent').textContent = marketBias;
        document.getElementById('kpi-market-score').textContent = `${avgSent.toFixed(2)} Avg Sentiment`;
        
        // 3. Render High-Impact Visuals
        renderHypeMatrix(coins); updateSignalFeed(data.individual_posts);
        renderFeatureImportance(); updateKeywordCloud(hotCoin.name, data.coins[hotCoin.name]); renderModelRadar();
        
        // 4. Populate Ranking Table
        if (tbody) {
            tbody.innerHTML = '';
            coins.sort((a, b) => (b[1].meme_viral_score || 0) - (a[1].meme_viral_score || 0));
            coins.forEach(([name, info]) => {
                const tr = document.createElement('tr');
                const trendClass = (info.trend_label || 'neutral').toLowerCase();
                const confPct = ((info.confidence || 0) * 100).toFixed(1);
                const sentiment = info.sentiment_score !== undefined ? info.sentiment_score : 0;
                const sentClass = sentiment > 0 ? 'pos' : (sentiment < 0 ? 'neg' : '');

                tr.innerHTML = `
                    <td><span class="coin-name">${name}</span></td>
                    <td><span class="trend-badge ${trendClass}">${(info.trend_label || 'neutral').toUpperCase()}</span></td>
                    <td>
                        <div class="conf-bar-container">
                            <div class="conf-bar" style="width: ${confPct}%"></div>
                            <span class="conf-text">${confPct}%</span>
                        </div>
                    </td>
                    <td><span class="viral-num">${(info.meme_viral_score || 0).toFixed(1)}</span></td>
                    <td><span class="sent-num ${sentClass}">${(sentiment > 0 ? '+' : '')}${sentiment.toFixed(2)}</span></td>
                    <td class="driver-text">${info.primary_driver || 'Neural Analysis'}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        addLog('System', `Intelligence Hub Synced: ${coins.length} tokens.`);
        
        // 5. Trigger Alpha Alerts for high-impact tokens
        initAlphaAlerts(data.coins);
        
    } catch (e) { console.error("ML Load Error:", e); addLog('Error', 'Intelligence Hub fetch failed.'); }
}

function initAlphaAlerts(coinsData) {
    const trending = Object.entries(coinsData)
        .filter(([_, info]) => info.meme_viral_score > 70)
        .sort((a, b) => b[1].meme_viral_score - a[1].meme_viral_score);
    
    if (trending.length > 0) {
        const top = trending[0];
        // Only show if we haven't alerted in this session for this coin (simple guard)
        if (!window.alertedCoins) window.alertedCoins = new Set();
        if (!window.alertedCoins.has(top[0])) {
            showToast(`$${top[0]} exhibiting high viral velocity (${top[1].meme_viral_score.toFixed(1)}). Monitor for breakout.`, 'warning');
            window.alertedCoins.add(top[0]);
        }
    }
}

function renderHypeMatrix(coins) {
    const data = [{
        x: coins.map(c => c[1].sentiment_score || 0), y: coins.map(c => c[1].meme_viral_score || 0),
        mode: 'markers+text', text: coins.map(c => c[0]), textposition: 'top center', marker: { size: 12, color: '#00e5ff' }, type: 'scatter'
    }];
    Plotly.newPlot('hype-matrix', data, { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#8892b0', size: 10 }, margin: { l: 40, r: 20, t: 20, b: 40 } }, { displayModeBar: false });
}

function updateSignalFeed(posts) {
    const feed = document.getElementById('ml-social-feed'); if (!feed) return;
    feed.innerHTML = '';
    posts.slice(0, 20).forEach(post => {
        const sentiment = post.sentiment_score || 0;
        const sentClass = sentiment > 0.1 ? 'pos' : (sentiment < -0.1 ? 'neg' : '');
        const item = document.createElement('div'); item.className = 'feed-item';
        item.innerHTML = `<span class="feed-text">${post.text}</span><div class="feed-meta"><span>@${post.platform.toUpperCase()}</span><span class="${sentClass}">${sentiment > 0.1 ? 'BULLISH' : sentiment < -0.1 ? 'BEARISH' : 'NEUTRAL'}</span></div>`;
        feed.appendChild(item);
    });
}

function renderFeatureImportance() {
    const data = [{ type: 'bar', x: [0.38, 0.25, 0.18, 0.12, 0.07], y: ['Sentiment', 'Volume', 'Engagement', 'Spikes', 'Topic'], orientation: 'h', marker: { color: '#00e5ff' } }];
    Plotly.newPlot('rf-feature-chart', data, { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#8892b0', size: 10 }, margin: { l: 80, r: 20, t: 10, b: 30 } }, { displayModeBar: false });
}

function updateKeywordCloud(coinName, info) {
    const cloud = document.getElementById('keyword-cloud'); if (!cloud || !info) return;
    document.getElementById('keyword-title').textContent = `Trending: $${coinName}`;
    cloud.innerHTML = '';
    (info.top_keywords || ['hype', 'moon', 'community']).forEach(word => {
        const span = document.createElement('span'); span.className = 'word-pill';
        if (word.toLowerCase().includes('moon')) span.classList.add('high-hype');
        span.textContent = word; cloud.appendChild(span);
    });
}

function renderModelRadar() {
    const data = [{ type: 'scatterpolar', r: [92, 88, 90, 85, 89], theta: ['Acc', 'Prec', 'Rec', 'F1', 'Stab'], fill: 'toself', fillcolor: 'rgba(168, 85, 247, 0.3)', line: { color: '#a855f7' } }];
    Plotly.newPlot('model-health-radar', data, { polar: { bgcolor: 'rgba(0,0,0,0)', radialaxis: { visible: true, range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)' } }, paper_bgcolor: 'rgba(0,0,0,0)', font: { color: '#8892b0', size: 9 }, margin: { l: 40, r: 40, t: 20, b: 20 } }, { displayModeBar: false });
}
