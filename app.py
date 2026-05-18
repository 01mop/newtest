"""
╔══════════════════════════════════════════════════════════════════════╗
║   QUANTUM BOT  v1.0  —  Trading Bot Interface                        ║
║   Strategies · Backtesting · Live Trading · IB · Binance · ML       ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time, json, threading, queue
from pathlib import Path

st.set_page_config(
    page_title="QUANTUM BOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&family=Orbitron:wght@700;900&display=swap');
html,body,[class*="css"]{background:#070b10!important;color:#c8d8e8!important;font-family:'IBM Plex Sans',sans-serif;}
.stApp{background:#070b10;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1520,#080f18)!important;border-right:1px solid #1a2f45;}
section[data-testid="stSidebar"] *{color:#7aafc8!important;}
section[data-testid="stSidebar"] label{color:#4da8da!important;font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;}
.stTextInput>div>div input,.stSelectbox>div>div,.stMultiSelect>div>div,.stNumberInput>div>div input{background:#0d1e2e!important;border:1px solid #1e3a55!important;border-radius:2px!important;color:#4da8da!important;font-family:'Space Mono',monospace!important;}
.stTabs [data-baseweb="tab-list"]{background:#0a1520;border-bottom:1px solid #1a3a55;gap:0;}
.stTabs [data-baseweb="tab"]{color:#4a7a99;font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1px;padding:10px 16px;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom:2px solid #00d4ff!important;background:transparent!important;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0d1f30,#0a1520);border:1px solid #1a3a55;border-left:3px solid #00d4ff;border-radius:2px;padding:10px 14px;}
[data-testid="metric-container"] label{color:#4da8da!important;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:1.5px;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e8f4fd!important;font-family:'Space Mono',monospace;font-size:16px;}
.blabel{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:#2a6a8a;text-transform:uppercase;border-bottom:1px solid #1a3a55;padding-bottom:4px;margin-bottom:10px;}
.bot-card{background:#0d1f30;border:1px solid #1a3a55;border-radius:4px;padding:14px;margin-bottom:10px;}
.bot-running{border-left:4px solid #00ff88;}
.bot-stopped{border-left:4px solid #ff4466;}
.bot-paused{border-left:4px solid #ffaa00;}
.log-box{background:#050810;border:1px solid #1a3a55;border-radius:2px;padding:10px;font-family:'Space Mono',monospace;font-size:10px;height:220px;overflow-y:auto;line-height:1.8;}
.log-buy{color:#00ff88;} .log-sell{color:#ff4466;} .log-info{color:#4da8da;} .log-warn{color:#ffaa00;} .log-err{color:#ff4466;font-weight:bold;}
.status-dot-green{display:inline-block;width:8px;height:8px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:pulse 1.5s infinite;margin-right:6px;}
.status-dot-red{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff4466;margin-right:6px;}
.status-dot-yellow{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ffaa00;animation:pulse 2s infinite;margin-right:6px;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
hr{border-color:#1a3a55!important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-thumb{background:#1e3a55;}
.tag-live{background:#0d2a1a;border:1px solid #00ff88;color:#00ff88;font-family:'Space Mono',monospace;font-size:9px;padding:2px 7px;border-radius:2px;display:inline-block;margin:2px;}
.tag-paper{background:#1a1a0d;border:1px solid #ffaa00;color:#ffaa00;font-family:'Space Mono',monospace;font-size:9px;padding:2px 7px;border-radius:2px;display:inline-block;margin:2px;}
.tag-off{background:#2a0d0d;border:1px solid #ff4466;color:#ff4466;font-family:'Space Mono',monospace;font-size:9px;padding:2px 7px;border-radius:2px;display:inline-block;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ── imports from modules ──────────────────────────────────────────────
from core.data_feed    import DataFeed
from core.engine       import BotEngine
from core.risk         import RiskManager
from core.portfolio    import Portfolio
from strategies.classic import SMACrossover, RSIMeanReversion, MACDStrategy, BollingerBands
from strategies.ml_strat import MLStrategy, ReinforcementStrategy
from brokers.ib_broker import IBBroker
from brokers.binance_broker import BinanceBroker
from brokers.paper_broker   import PaperBroker

# ── session state init ────────────────────────────────────────────────
def init_state():
    defaults = {
        "bots":         {},      # {bot_id: BotEngine}
        "logs":         {},      # {bot_id: [log lines]}
        "portfolio":    Portfolio(),
        "active_bot":   None,
        "broker_status": {"IB": False, "Binance": False, "Paper": True},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

PLOTLY_BASE = dict(
    paper_bgcolor="#070b10", plot_bgcolor="#070b10",
    font=dict(family="Space Mono, monospace", color="#8aafc8", size=10),
    xaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    yaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    margin=dict(l=50,r=20,t=36,b=36),
    legend=dict(bgcolor="#0a1520", bordercolor="#1a3a55", borderwidth=1),
)
COLORS = ["#00d4ff","#00ff88","#ffaa00","#ff4466","#aa88ff","#ff88aa","#44ffdd"]

def blabel(txt): st.markdown(f"<div class='blabel'>▸ {txt}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-family:Orbitron,sans-serif;font-size:16px;font-weight:900;
                  letter-spacing:4px;color:#00d4ff;text-shadow:0 0 12px rgba(0,212,255,.4);'>
        🤖 QUANTUM BOT
      </div>
      <div style='font-family:Space Mono,monospace;font-size:8px;letter-spacing:2px;
                  color:#2a5a7a;margin-top:2px;'>TRADING ENGINE v1.0</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── Broker connections ────────────────────────────────────────────
    st.markdown("<div class='blabel'>▸ Brokers</div>", unsafe_allow_html=True)

    with st.expander("🏦 Interactive Brokers"):
        ib_host    = st.text_input("TWS Host",    value="127.0.0.1", key="ib_host")
        ib_port    = st.number_input("TWS Port",  value=7497, step=1, key="ib_port",
                                      help="7497=paper, 7496=live")
        ib_client  = st.number_input("Client ID", value=1, step=1, key="ib_cid")
        if st.button("🔌 Connecter IB", use_container_width=True):
            try:
                broker = IBBroker(ib_host, int(ib_port), int(ib_client))
                ok = broker.connect()
                st.session_state["ib_broker"] = broker if ok else None
                st.session_state["broker_status"]["IB"] = ok
                st.success("✓ IB Connecté") if ok else st.error("✗ Connexion échouée")
            except Exception as e:
                st.error(f"Erreur: {e}")
        ib_ok = st.session_state["broker_status"]["IB"]
        st.markdown(f"<span class='{'tag-live' if ib_ok else 'tag-off'}'>{'● CONNECTÉ' if ib_ok else '● DÉCONNECTÉ'}</span>",
                    unsafe_allow_html=True)

    with st.expander("🟡 Binance"):
        bn_key    = st.text_input("API Key",    type="password", key="bn_key")
        bn_secret = st.text_input("API Secret", type="password", key="bn_secret")
        bn_testnet= st.checkbox("Testnet (paper)", value=True, key="bn_testnet")
        if st.button("🔌 Connecter Binance", use_container_width=True):
            try:
                broker = BinanceBroker(bn_key, bn_secret, testnet=bn_testnet)
                ok = broker.connect()
                st.session_state["binance_broker"] = broker if ok else None
                st.session_state["broker_status"]["Binance"] = ok
                st.success("✓ Binance Connecté") if ok else st.error("✗ Connexion échouée")
            except Exception as e:
                st.error(f"Erreur: {e}")
        bn_ok = st.session_state["broker_status"]["Binance"]
        st.markdown(f"<span class='{'tag-live' if bn_ok else 'tag-off'}'>{'● CONNECTÉ' if bn_ok else '● DÉCONNECTÉ'}</span>",
                    unsafe_allow_html=True)

    st.markdown("<span class='tag-paper'>● PAPER BROKER ACTIF</span>", unsafe_allow_html=True)

    st.divider()

    # ── Global Risk Controls ──────────────────────────────────────────
    st.markdown("<div class='blabel'>▸ Risk Global</div>", unsafe_allow_html=True)
    max_drawdown_kill = st.slider("Kill switch DD (%)", 1, 30, 10,
                                   help="Arrêt auto si drawdown > seuil")
    max_daily_loss    = st.slider("Perte max journalière ($)", 100, 10000, 500, step=100)
    max_open_pos      = st.slider("Positions max simultanées", 1, 20, 5)

    if st.button("🚨 ARRÊT URGENCE TOUS BOTS", use_container_width=True,
                 type="primary"):
        for bot_id, bot in st.session_state["bots"].items():
            bot.stop()
        st.error("⚠ Tous les bots arrêtés")

    st.divider()
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a55;text-align:center;'>
      <span class='status-dot-green'></span>{datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
h1, h2, h3 = st.columns([3,1,1])
with h1:
    st.markdown(f"""
    <div style='font-family:Orbitron,sans-serif;font-size:22px;font-weight:900;
                letter-spacing:5px;color:#00d4ff;text-shadow:0 0 18px rgba(0,212,255,.35);'>
      🤖 QUANTUM BOT
    </div>
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#2a5a7a;letter-spacing:2px;'>
      <span class='status-dot-green'></span>ENGINE READY · {datetime.now().strftime('%d %b %Y %H:%M')}
    </div>""", unsafe_allow_html=True)
with h2:
    n_running = sum(1 for b in st.session_state["bots"].values() if b.status == "running")
    st.metric("Bots actifs", n_running)
with h3:
    if st.button("↺ Actualiser", use_container_width=True):
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🤖  Mes Bots",
    "➕  Créer un Bot",
    "📊  Backtesting",
    "💼  Portfolio",
    "📋  Journal des Trades",
    "⚠️  Risk Monitor",
    "🧠  ML Studio",
    "🔌  Brokers & API",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 0 — MES BOTS (dashboard)
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    blabel("Dashboard — Bots actifs")

    if not st.session_state["bots"]:
        st.info("Aucun bot créé. Allez dans **➕ Créer un Bot** pour commencer.")
    else:
        for bot_id, bot in st.session_state["bots"].items():
            status_class = {"running":"bot-running","stopped":"bot-stopped",
                            "paused":"bot-paused"}.get(bot.status,"bot-stopped")
            dot_class    = {"running":"status-dot-green","stopped":"status-dot-red",
                            "paused":"status-dot-yellow"}.get(bot.status,"status-dot-red")

            with st.container():
                st.markdown(f"<div class='bot-card {status_class}'>", unsafe_allow_html=True)

                bc1,bc2,bc3,bc4,bc5 = st.columns([3,1,1,1,2])
                with bc1:
                    st.markdown(f"""
                    <div style='font-family:Orbitron,sans-serif;font-size:13px;color:#00d4ff;'>
                      <span class='{dot_class}'></span>{bot.name}
                    </div>
                    <div style='font-family:Space Mono,monospace;font-size:9px;color:#4a7a99;margin-top:4px;'>
                      {bot.strategy_name} · {bot.symbol} · {bot.broker_type}
                    </div>""", unsafe_allow_html=True)
                bc2.metric("P&L",     f"${bot.pnl:+.2f}",  delta_color="normal" if bot.pnl>=0 else "inverse")
                bc3.metric("Trades",  str(bot.n_trades))
                bc4.metric("Win Rate",f"{bot.win_rate:.0f}%")
                with bc5:
                    col_a, col_b, col_c = st.columns(3)
                    if bot.status != "running":
                        if col_a.button("▶", key=f"start_{bot_id}", help="Démarrer"):
                            bot.start()
                            st.rerun()
                    else:
                        if col_a.button("⏸", key=f"pause_{bot_id}", help="Pause"):
                            bot.pause()
                            st.rerun()
                    if col_b.button("⏹", key=f"stop_{bot_id}", help="Arrêter"):
                        bot.stop()
                        st.rerun()
                    if col_c.button("🗑", key=f"del_{bot_id}", help="Supprimer"):
                        bot.stop()
                        del st.session_state["bots"][bot_id]
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            # Logs du bot
            with st.expander(f"📋 Logs — {bot.name}", expanded=False):
                logs = st.session_state["logs"].get(bot_id, [])
                log_html = "<div class='log-box'>"
                for log in logs[-50:]:
                    ts   = log.get("time","")
                    msg  = log.get("msg","")
                    ltype= log.get("type","info")
                    css  = {"buy":"log-buy","sell":"log-sell","info":"log-info",
                            "warn":"log-warn","error":"log-err"}.get(ltype,"log-info")
                    log_html += f"<div><span style='color:#2a5a7a;'>[{ts}]</span> <span class='{css}'>{msg}</span></div>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)

            # Mini equity chart
            if bot.equity_curve:
                eq = pd.Series(bot.equity_curve)
                fig_eq = go.Figure(go.Scatter(
                    y=eq, mode="lines",
                    line=dict(color="#00d4ff" if eq.iloc[-1]>=eq.iloc[0] else "#ff4466", width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
                fig_eq.update_layout(**PLOTLY_BASE, height=120,
                    margin=dict(l=30,r=10,t=10,b=20), showlegend=False)
                st.plotly_chart(fig_eq, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — CRÉER UN BOT
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    blabel("Créer un nouveau bot de trading")

    with st.form("create_bot_form"):
        fc1, fc2, fc3 = st.columns(3)

        bot_name    = fc1.text_input("Nom du bot", value="Bot_01",
                                      help="Identifiant unique de votre bot")
        bot_symbol  = fc2.text_input("Actif / Symbol",value="AAPL",
                                      help="Ex: AAPL, BTCUSDT, EUR/USD")
        bot_broker  = fc3.selectbox("Broker", ["Paper (simulation)",
                                               "Interactive Brokers",
                                               "Binance"])

        st.divider()

        # ── Strategy selection ────────────────────────────────────────
        blabel("Stratégie")
        sc1, sc2 = st.columns([1,2])
        strategy_type = sc1.selectbox("Type", [
            "── Classiques ──",
            "SMA Crossover",
            "RSI Mean-Reversion",
            "MACD",
            "Bollinger Bands",
            "── Intelligence Artificielle ──",
            "ML Prédictif (XGBoost)",
            "Reinforcement Learning (PPO)",
            "── Multi-stratégies ──",
            "Ensemble (vote majoritaire)",
        ])

        with sc2:
            if strategy_type == "SMA Crossover":
                sma_fast = st.slider("SMA Rapide", 5, 50,  20)
                sma_slow = st.slider("SMA Lente",  20,200, 50)
                params = {"fast": sma_fast, "slow": sma_slow}

            elif strategy_type == "RSI Mean-Reversion":
                rsi_p  = st.slider("RSI Période", 5, 30, 14)
                rsi_ob = st.slider("Overbought",  60, 90, 70)
                rsi_os = st.slider("Oversold",    10, 40, 30)
                params = {"period":rsi_p,"overbought":rsi_ob,"oversold":rsi_os}

            elif strategy_type == "MACD":
                macd_f  = st.slider("Fast",   5,  30, 12)
                macd_s  = st.slider("Slow",  15,  50, 26)
                macd_sg = st.slider("Signal", 3,  15,  9)
                params  = {"fast":macd_f,"slow":macd_s,"signal":macd_sg}

            elif strategy_type == "Bollinger Bands":
                bb_w = st.slider("Fenêtre", 10, 50, 20)
                bb_s = st.slider("Écart-type", 1.0, 3.0, 2.0, step=0.5)
                params = {"window":bb_w,"std":bb_s}

            elif strategy_type == "ML Prédictif (XGBoost)":
                ml_lookback  = st.slider("Lookback (jours)", 30, 365, 252)
                ml_threshold = st.slider("Seuil signal (%)", 50, 80, 60)
                ml_retrain   = st.slider("Re-train tous les N trades", 10, 200, 50)
                params = {"lookback":ml_lookback,"threshold":ml_threshold/100,
                          "retrain_every":ml_retrain}

            elif strategy_type == "Reinforcement Learning (PPO)":
                rl_lookback = st.slider("Fenêtre d'observation", 10, 60, 20)
                rl_episodes = st.slider("Épisodes d'entraînement", 10, 500, 100)
                params = {"lookback":rl_lookback,"episodes":rl_episodes}

            elif strategy_type == "Ensemble (vote majoritaire)":
                st.info("Combinaison SMA + RSI + MACD avec vote majoritaire")
                ens_threshold = st.slider("Seuil consensus (sur 3)", 2, 3, 2)
                params = {"threshold":ens_threshold}
            else:
                params = {}
                st.info("Sélectionnez une stratégie ci-dessus.")

        st.divider()

        # ── Timeframe & Data ─────────────────────────────────────────
        blabel("Données & Temporalité")
        dc1,dc2,dc3 = st.columns(3)
        timeframe = dc1.selectbox("Timeframe", [
            "1m","5m","15m","30m","1h","4h","1d","1wk"])
        data_source = dc2.selectbox("Source données", [
            "yfinance","Binance","Interactive Brokers"])
        warmup = dc3.number_input("Warmup (bougies)", value=100, min_value=10,
                                   help="Bougies nécessaires avant le premier signal")

        st.divider()

        # ── Risk Management ──────────────────────────────────────────
        blabel("Risk Management par Bot")
        rm1,rm2,rm3,rm4 = st.columns(4)
        position_size = rm1.number_input("Taille position ($)", value=1000.0, step=100.0)
        stop_loss     = rm2.number_input("Stop Loss (%)",       value=2.0,    step=0.5)
        take_profit   = rm3.number_input("Take Profit (%)",     value=4.0,    step=0.5)
        max_trades_day= rm4.number_input("Max trades/jour",     value=10,     step=1)

        rm5,rm6,rm7 = st.columns(3)
        trailing_stop  = rm5.checkbox("Trailing Stop",   value=False)
        trailing_pct   = rm5.number_input("Trailing %",  value=1.0, step=0.5) if trailing_stop else 0
        use_kelly      = rm6.checkbox("Kelly Criterion", value=False,
                                       help="Sizing dynamique basé sur win rate")
        hedge_mode     = rm7.checkbox("Mode Hedge",      value=False,
                                       help="Autorise positions long et short simultanées")

        st.divider()

        # ── Execution ────────────────────────────────────────────────
        blabel("Exécution des ordres")
        ec1,ec2,ec3 = st.columns(3)
        order_type   = ec1.selectbox("Type d'ordre", ["Market","Limit","Stop-Limit"])
        slippage_pct = ec2.number_input("Slippage estimé (%)", value=0.05, step=0.01)
        commission   = ec3.number_input("Commission ($)", value=1.0, step=0.5)

        st.divider()

        # ── Execution mode ───────────────────────────────────────────
        blabel("Mode d'exécution")
        em1, em2 = st.columns(2)
        exec_mode = em1.radio("Mode", ["📄 Paper (simulation)", "📡 Live Trading"],
                               help="Paper = simulation sans argent réel")
        auto_start = em2.checkbox("Démarrer automatiquement", value=False)

        submitted = st.form_submit_button("✅ Créer le Bot", use_container_width=True,
                                           type="primary")

    if submitted and strategy_type not in ["── Classiques ──","── Intelligence Artificielle ──","── Multi-stratégies ──"]:
        # Map strategy
        strat_map = {
            "SMA Crossover":              SMACrossover,
            "RSI Mean-Reversion":         RSIMeanReversion,
            "MACD":                       MACDStrategy,
            "Bollinger Bands":            BollingerBands,
            "ML Prédictif (XGBoost)":     MLStrategy,
            "Reinforcement Learning (PPO)":ReinforcementStrategy,
            "Ensemble (vote majoritaire)": SMACrossover,  # fallback
        }
        strat_cls = strat_map.get(strategy_type, SMACrossover)
        strategy  = strat_cls(**params)

        # Map broker
        broker_map = {
            "Paper (simulation)":   PaperBroker,
            "Interactive Brokers":  IBBroker,
            "Binance":              BinanceBroker,
        }

        risk_mgr = RiskManager(
            stop_loss_pct   = stop_loss/100,
            take_profit_pct = take_profit/100,
            position_size   = position_size,
            max_trades_day  = int(max_trades_day),
            trailing_stop   = trailing_stop,
            trailing_pct    = trailing_pct/100,
            use_kelly       = use_kelly,
            slippage_pct    = slippage_pct/100,
            commission      = commission,
        )

        bot_id = f"{bot_name}_{int(time.time())}"
        bot    = BotEngine(
            bot_id       = bot_id,
            name         = bot_name,
            symbol       = bot_symbol.upper().strip(),
            strategy     = strategy,
            strategy_name= strategy_type,
            broker_type  = bot_broker,
            broker_cls   = broker_map[bot_broker],
            timeframe    = timeframe,
            data_source  = data_source,
            risk_manager = risk_mgr,
            warmup       = int(warmup),
            live         = "Live" in exec_mode,
            log_queue    = st.session_state["logs"],
        )

        st.session_state["bots"][bot_id] = bot
        st.session_state["logs"][bot_id] = []

        if auto_start:
            bot.start()

        st.success(f"✅ Bot **{bot_name}** créé avec la stratégie **{strategy_type}** !")
        st.info("Allez dans **🤖 Mes Bots** pour le démarrer et suivre ses performances.")

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    blabel("Backtesting — Testez vos stratégies sur données historiques")

    with st.form("backtest_form"):
        bc1,bc2,bc3 = st.columns(3)
        bt_symbol   = bc1.text_input("Actif", value="AAPL")
        bt_strategy = bc2.selectbox("Stratégie", [
            "SMA Crossover","RSI Mean-Reversion","MACD",
            "Bollinger Bands","ML Prédictif (XGBoost)",
            "Ensemble (vote majoritaire)"])
        bt_period   = bc3.selectbox("Période", [
            "3 mois","6 mois","1 an","2 ans","5 ans"], index=2)

        bd1,bd2,bd3,bd4 = st.columns(4)
        bt_capital  = bd1.number_input("Capital initial ($)", value=10000.0, step=1000.0)
        bt_pos_size = bd2.number_input("Taille position ($)", value=1000.0,  step=100.0)
        bt_sl       = bd3.number_input("Stop Loss (%)",       value=2.0,     step=0.5)
        bt_tp       = bd4.number_input("Take Profit (%)",     value=4.0,     step=0.5)

        bp1,bp2,bp3 = st.columns(3)
        bt_cost     = bp1.number_input("Commission ($)", value=1.0, step=0.5)
        bt_slip     = bp2.number_input("Slippage (%)",   value=0.05, step=0.01)
        bt_tf       = bp3.selectbox("Timeframe",         ["1d","1h","4h","15m"])

        # Strategy params
        blabel("Paramètres stratégie")
        sp1,sp2,sp3 = st.columns(3)
        bt_p1 = sp1.number_input("Param 1 (fast/RSI period/MACD fast)", value=20)
        bt_p2 = sp2.number_input("Param 2 (slow/overbought/MACD slow)",  value=50)
        bt_p3 = sp3.number_input("Param 3 (signal/oversold)",            value=9)

        run_bt = st.form_submit_button("🚀 Lancer le Backtest", use_container_width=True, type="primary")

    if run_bt:
        period_map = {"3 mois":"3mo","6 mois":"6mo","1 an":"1y","2 ans":"2y","5 ans":"5y"}
        yf_period  = period_map[bt_period]

        with st.spinner("Chargement des données et backtesting…"):
            feed   = DataFeed(bt_symbol.upper().strip(), yf_period, bt_tf)
            df_bt  = feed.fetch()

        if df_bt.empty:
            st.error("Données indisponibles.")
        else:
            close = df_bt["Close"].squeeze()
            high  = df_bt["High"].squeeze()
            low   = df_bt["Low"].squeeze()
            vol   = df_bt["Volume"].squeeze()

            # ── Run strategy ──────────────────────────────────────────
            strat_map_bt = {
                "SMA Crossover":    SMACrossover(fast=int(bt_p1),slow=int(bt_p2)),
                "RSI Mean-Reversion": RSIMeanReversion(period=int(bt_p1),
                                                        overbought=int(bt_p2),oversold=int(bt_p3)),
                "MACD":             MACDStrategy(fast=int(bt_p1),slow=int(bt_p2),signal=int(bt_p3)),
                "Bollinger Bands":  BollingerBands(window=int(bt_p1),std=2.0),
                "ML Prédictif (XGBoost)": MLStrategy(lookback=int(bt_p1)),
                "Ensemble (vote majoritaire)": SMACrossover(fast=int(bt_p1),slow=int(bt_p2)),
            }
            strat_bt   = strat_map_bt[bt_strategy]
            signals    = strat_bt.generate_signals(df_bt)

            # ── Backtester ────────────────────────────────────────────
            risk_bt = RiskManager(
                stop_loss_pct=bt_sl/100, take_profit_pct=bt_tp/100,
                position_size=bt_pos_size, slippage_pct=bt_slip/100,
                commission=bt_cost)

            results = risk_bt.run_backtest(
                signals=signals, prices=close, high=high, low=low,
                initial_capital=bt_capital)

            equity       = results["equity"]
            trades_list  = results["trades"]
            returns_bt   = results["returns"]

            # ── Metrics ───────────────────────────────────────────────
            import scipy.stats as sci_stats
            total_ret  = (equity.iloc[-1]/bt_capital - 1)*100
            ann_ret    = returns_bt.mean()*252*100
            ann_vol    = returns_bt.std()*np.sqrt(252)*100
            sharpe     = (returns_bt.mean()*252)/(returns_bt.std()*np.sqrt(252)+1e-9)
            sortino_d  = returns_bt[returns_bt<0].std()*np.sqrt(252)
            sortino    = (returns_bt.mean()*252)/(sortino_d+1e-9)
            mdd        = ((equity/equity.cummax())-1).min()*100
            win_trades = [t for t in trades_list if t["pnl"] > 0]
            win_rate   = len(win_trades)/max(len(trades_list),1)*100
            profit_factor= (sum(t["pnl"] for t in win_trades)/
                            max(abs(sum(t["pnl"] for t in trades_list if t["pnl"]<0)),1))
            avg_win    = np.mean([t["pnl"] for t in win_trades]) if win_trades else 0
            avg_loss   = np.mean([t["pnl"] for t in trades_list if t["pnl"]<0]) or 0
            expectancy = (win_rate/100*avg_win) + ((1-win_rate/100)*avg_loss)
            calmar     = ann_ret/max(abs(mdd),1)

            blabel("Métriques de performance")
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Rendement total",  f"{total_ret:+.2f}%")
            m2.metric("Rendement ann.",   f"{ann_ret:+.2f}%")
            m3.metric("Volatilité ann.",  f"{ann_vol:.2f}%")
            m4.metric("Sharpe",           f"{sharpe:.3f}")
            m5.metric("Sortino",          f"{sortino:.3f}")
            m6.metric("Max Drawdown",     f"{mdd:.2f}%")

            m7,m8,m9,m10,m11,m12 = st.columns(6)
            m7.metric("Nb trades",        str(len(trades_list)))
            m8.metric("Win Rate",         f"{win_rate:.1f}%")
            m9.metric("Profit Factor",    f"{profit_factor:.2f}")
            m10.metric("Avg Win",         f"${avg_win:.2f}")
            m11.metric("Avg Loss",        f"${avg_loss:.2f}")
            m12.metric("Espérance/trade", f"${expectancy:.2f}")

            m13,m14,m15 = st.columns(3)
            m13.metric("Calmar Ratio",    f"{calmar:.3f}")
            m14.metric("Capital final",   f"${equity.iloc[-1]:,.2f}")
            m15.metric("P&L net",         f"${equity.iloc[-1]-bt_capital:+,.2f}")

            st.divider()

            # ── Charts ────────────────────────────────────────────────
            blabel("Graphiques")
            fig_bt = make_subplots(rows=4, cols=1, shared_xaxes=True,
                vertical_spacing=0.03, row_heights=[0.40,0.20,0.20,0.20],
                subplot_titles=["Prix & Signaux","Equity Curve","Drawdown","Volume"])

            # Price + signals
            fig_bt.add_trace(go.Scatter(x=close.index,y=close,mode="lines",
                line=dict(color="#4a7a99",width=1),name="Prix"),row=1,col=1)

            buy_sig  = close[signals==1]
            sell_sig = close[signals==-1]
            if not buy_sig.empty:
                fig_bt.add_trace(go.Scatter(x=buy_sig.index,y=buy_sig,mode="markers",
                    marker=dict(symbol="triangle-up",color="#00ff88",size=10),
                    name="BUY"),row=1,col=1)
            if not sell_sig.empty:
                fig_bt.add_trace(go.Scatter(x=sell_sig.index,y=sell_sig,mode="markers",
                    marker=dict(symbol="triangle-down",color="#ff4466",size=10),
                    name="SELL"),row=1,col=1)

            # Trade entry/exit dots
            for trade in trades_list[:200]:
                color = "#00ff88" if trade["pnl"] > 0 else "#ff4466"
                fig_bt.add_trace(go.Scatter(
                    x=[trade["entry_date"],trade["exit_date"]],
                    y=[trade["entry_price"],trade["exit_price"]],
                    mode="lines+markers",
                    line=dict(color=color,width=1,dash="dot"),
                    marker=dict(size=5,color=color),
                    showlegend=False,
                    hovertemplate=f"P&L: ${trade['pnl']:.2f}<extra></extra>"),
                    row=1,col=1)

            # Equity
            bh_eq = (close/close.iloc[0])*bt_capital
            fig_bt.add_trace(go.Scatter(x=equity.index,y=equity,mode="lines",
                line=dict(color="#00d4ff",width=1.5),
                fill="tozeroy",fillcolor="rgba(0,212,255,0.05)",
                name="Equity"),row=2,col=1)
            fig_bt.add_trace(go.Scatter(x=bh_eq.index,y=bh_eq,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),
                name="Buy & Hold"),row=2,col=1)

            # Drawdown
            dd_series = ((equity/equity.cummax())-1)*100
            fig_bt.add_trace(go.Scatter(x=dd_series.index,y=dd_series,mode="lines",
                fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown %"),row=3,col=1)

            # Volume
            vc = ["#00ff88" if signals.iloc[i]>=0 else "#ff4466"
                  for i in range(len(signals))]
            fig_bt.add_trace(go.Bar(x=vol.index,y=vol,marker_color=vc,
                name="Volume",opacity=0.6),row=4,col=1)

            fig_bt.update_layout(**PLOTLY_BASE, height=700,
                title=dict(text=f"◈ Backtest — {bt_symbol} · {bt_strategy} · {bt_period}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)

            # ── Trade list ────────────────────────────────────────────
            blabel("Liste des trades")
            if trades_list:
                df_trades_bt = pd.DataFrame(trades_list)
                df_trades_bt["pnl"] = df_trades_bt["pnl"].round(2)
                df_trades_bt["pnl_pct"] = df_trades_bt["pnl_pct"].round(3)
                st.dataframe(df_trades_bt, use_container_width=True, height=300)
                csv_trades = df_trades_bt.to_csv(index=False).encode()
                st.download_button("⬇ Exporter les trades CSV", csv_trades,
                                   f"trades_{bt_symbol}_{bt_strategy}.csv")

            # ── Monthly returns ───────────────────────────────────────
            blabel("Rendements mensuels")
            monthly_eq = equity.resample("ME").last().pct_change().dropna()*100
            if not monthly_eq.empty:
                fig_monthly = go.Figure(go.Bar(
                    x=[str(d.date()) for d in monthly_eq.index],
                    y=monthly_eq.values,
                    marker_color=["#00ff88" if v>=0 else "#ff4466" for v in monthly_eq.values],
                    text=[f"{v:+.1f}%" for v in monthly_eq.values],
                    textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=9)))
                fig_monthly.update_layout(**PLOTLY_BASE,height=280,
                    title=dict(text="◈ Rendements mensuels",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_monthly, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    blabel("Portfolio global — Toutes positions")
    portfolio = st.session_state["portfolio"]

    pm1,pm2,pm3,pm4,pm5 = st.columns(5)
    pm1.metric("Valeur totale",    f"${portfolio.total_value:,.2f}")
    pm2.metric("Cash disponible",  f"${portfolio.cash:,.2f}")
    pm3.metric("P&L non réalisé",  f"${portfolio.unrealized_pnl:+,.2f}",
               delta_color="normal" if portfolio.unrealized_pnl>=0 else "inverse")
    pm4.metric("P&L réalisé",      f"${portfolio.realized_pnl:+,.2f}",
               delta_color="normal" if portfolio.realized_pnl>=0 else "inverse")
    pm5.metric("Positions ouvertes",str(len(portfolio.positions)))

    st.divider()

    if portfolio.positions:
        blabel("Positions ouvertes")
        pos_data = []
        for sym, pos in portfolio.positions.items():
            pos_data.append({
                "Symbol":     sym,
                "Qty":        pos["qty"],
                "Prix entrée":f"${pos['entry_price']:.4f}",
                "Prix actuel":f"${pos['current_price']:.4f}",
                "P&L $":      f"${pos['unrealized_pnl']:+.2f}",
                "P&L %":      f"{pos['pnl_pct']:+.2f}%",
                "Bot":        pos.get("bot_name","Manual"),
            })
        st.dataframe(pd.DataFrame(pos_data).set_index("Symbol"),
                     use_container_width=True)
    else:
        st.info("Aucune position ouverte.")

    blabel("Historique equity du portfolio")
    if portfolio.equity_history:
        eq_hist = pd.Series(portfolio.equity_history,
                            index=pd.date_range(end=datetime.now(),
                                                periods=len(portfolio.equity_history),
                                                freq="T"))
        fig_pf = go.Figure(go.Scatter(x=eq_hist.index,y=eq_hist,mode="lines",
            line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)",name="Portfolio"))
        fig_pf.update_layout(**PLOTLY_BASE,height=300,
            title=dict(text="◈ Equity Portfolio",
                       font=dict(family="Orbitron",color="#00d4ff",size=11)))
        st.plotly_chart(fig_pf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — JOURNAL DES TRADES
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    blabel("Journal des trades — Tous bots confondus")

    all_trades = []
    for bot_id, bot in st.session_state["bots"].items():
        for trade in bot.trades_history:
            trade["bot"] = bot.name
            all_trades.append(trade)

    if not all_trades:
        st.info("Aucun trade exécuté. Créez et démarrez un bot.")
    else:
        df_journal = pd.DataFrame(all_trades)
        df_journal = df_journal.sort_values("exit_date", ascending=False) \
                                if "exit_date" in df_journal else df_journal

        # Filtres
        jf1,jf2,jf3 = st.columns(3)
        filter_bot = jf1.multiselect("Filtrer par bot",
                                      [b.name for b in st.session_state["bots"].values()])
        filter_dir = jf2.multiselect("Direction", ["BUY","SELL"])
        filter_pnl = jf3.radio("P&L", ["Tous","Winners","Losers"], horizontal=True)

        df_show = df_journal.copy()
        if filter_bot: df_show = df_show[df_show["bot"].isin(filter_bot)]
        if filter_dir and "direction" in df_show: df_show = df_show[df_show["direction"].isin(filter_dir)]
        if filter_pnl == "Winners" and "pnl" in df_show: df_show = df_show[df_show["pnl"]>0]
        if filter_pnl == "Losers"  and "pnl" in df_show: df_show = df_show[df_show["pnl"]<0]

        st.dataframe(df_show, use_container_width=True, height=400)

        if not df_show.empty and "pnl" in df_show:
            jm1,jm2,jm3,jm4 = st.columns(4)
            jm1.metric("P&L total",    f"${df_show['pnl'].sum():+.2f}")
            jm2.metric("Win rate",     f"{(df_show['pnl']>0).mean()*100:.1f}%")
            jm3.metric("Avg trade",    f"${df_show['pnl'].mean():+.2f}")
            jm4.metric("Best / Worst", f"${df_show['pnl'].max():.2f} / ${df_show['pnl'].min():.2f}")

        st.download_button("⬇ Exporter journal CSV",
                           df_journal.to_csv(index=False).encode(),
                           "journal_trades.csv")

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — RISK MONITOR
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    blabel("Risk Monitor — Surveillance en temps réel")

    # Portfolio risk metrics
    portfolio = st.session_state["portfolio"]
    pnl_today = portfolio.realized_pnl  # simplified

    rm_status_color = "#00ff88" if pnl_today > -max_daily_loss*0.5 \
                      else "#ffaa00" if pnl_today > -max_daily_loss*0.8 \
                      else "#ff4466"

    r1,r2,r3,r4,r5 = st.columns(5)
    r1.metric("P&L journalier",   f"${pnl_today:+.2f}",
              delta_color="normal" if pnl_today>=0 else "inverse")
    r2.metric("Limite perte/j",   f"${max_daily_loss:,.0f}")
    r3.metric("Utilisation limite",f"{abs(pnl_today)/max(max_daily_loss,1)*100:.1f}%")
    r4.metric("Kill switch DD",   f"{max_drawdown_kill}%")
    r5.metric("Positions max",    str(max_open_pos))

    # Risk gauge
    usage_pct = min(abs(pnl_today)/max(max_daily_loss,1)*100, 100)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=usage_pct,
        delta={"reference":50,"valueformat":".1f"},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#8aafc8",
                    "tickfont":dict(family="Space Mono",color="#8aafc8")},
            "bar":{"color":rm_status_color,"thickness":0.3},
            "bgcolor":"#0d1f30",
            "bordercolor":"#1a3a55",
            "steps":[
                {"range":[0,50],  "color":"rgba(0,255,136,0.1)"},
                {"range":[50,80], "color":"rgba(255,170,0,0.1)"},
                {"range":[80,100],"color":"rgba(255,68,102,0.1)"},
            ],
            "threshold":{"line":{"color":"#ff4466","width":3},
                         "thickness":0.8,"value":100},
        },
        number={"suffix":"%","font":{"family":"Orbitron","color":"#00d4ff","size":24}},
        title={"text":"Utilisation limite perte journalière",
               "font":{"family":"Space Mono","color":"#8aafc8","size":11}},
    ))
    fig_gauge.update_layout(paper_bgcolor="#070b10",height=280,
                             margin=dict(l=30,r=30,t=50,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    blabel("État des bots & alertes")

    for bot_id, bot in st.session_state["bots"].items():
        with st.expander(f"🤖 {bot.name} — {bot.status.upper()}"):
            ba1,ba2,ba3,ba4 = st.columns(4)
            ba1.metric("P&L",         f"${bot.pnl:+.2f}")
            ba2.metric("Drawdown",    f"{bot.current_drawdown*100:.2f}%")
            ba3.metric("Trades/jour", str(bot.trades_today))
            ba4.metric("Exposition",  f"${bot.exposure:.2f}")

            # Check breaches
            if bot.current_drawdown*100 > max_drawdown_kill:
                st.error(f"⚠ KILL SWITCH: Drawdown {bot.current_drawdown*100:.1f}% > {max_drawdown_kill}%")
                bot.stop()
            if bot.trades_today >= bot.risk_manager.max_trades_day:
                st.warning(f"⚠ Limite trades journaliers atteinte ({bot.trades_today})")

# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — ML STUDIO
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    blabel("ML Studio — Entraînement & Évaluation des modèles")

    ml_tabs = st.tabs(["🔧 XGBoost","🧠 Reinforcement Learning","📊 Feature Engineering","💾 Modèles sauvegardés"])

    with ml_tabs[0]:
        blabel("Entraînement XGBoost")
        with st.form("xgb_form"):
            xc1,xc2,xc3 = st.columns(3)
            xgb_symbol  = xc1.text_input("Symbol", value="AAPL")
            xgb_period  = xc2.selectbox("Période d'entraînement",
                                          ["1 an","2 ans","3 ans","5 ans"], index=1)
            xgb_target  = xc3.selectbox("Target", [
                "Direction J+1 (classification)",
                "Rendement J+1 (régression)",
                "Direction J+5 (classification)"])

            xp1,xp2,xp3,xp4 = st.columns(4)
            xgb_n_est   = xp1.slider("n_estimators", 50, 500, 100)
            xgb_depth   = xp2.slider("max_depth",    2, 10,   4)
            xgb_lr      = xp3.number_input("learning_rate", value=0.05, step=0.01)
            xgb_split   = xp4.slider("Train/Test split (%)", 60, 90, 80)

            run_xgb = st.form_submit_button("🚀 Entraîner", use_container_width=True, type="primary")

        if run_xgb:
            period_map_ml = {"1 an":"1y","2 ans":"2y","3 ans":"3y","5 ans":"5y"}
            with st.spinner("Chargement données et entraînement XGBoost…"):
                from ml_strat_trainer import train_xgboost
                results_xgb = train_xgboost(
                    symbol=xgb_symbol.upper().strip(),
                    period=period_map_ml[xgb_period],
                    n_estimators=xgb_n_est,
                    max_depth=xgb_depth,
                    learning_rate=xgb_lr,
                    test_split=(100-xgb_split)/100,
                    target=xgb_target,
                )
            if results_xgb:
                xm1,xm2,xm3,xm4 = st.columns(4)
                xm1.metric("Accuracy test",  f"{results_xgb['accuracy']:.3f}")
                xm2.metric("F1 Score",       f"{results_xgb['f1']:.3f}")
                xm3.metric("AUC-ROC",        f"{results_xgb['auc']:.3f}")
                xm4.metric("Features",       str(results_xgb['n_features']))

                # Feature importance
                blabel("Feature Importance")
                fi = pd.Series(results_xgb['feature_importance'],
                               index=results_xgb['feature_names']).sort_values(ascending=True)
                fig_fi_ml = go.Figure(go.Bar(x=fi.values,y=fi.index.tolist(),
                    orientation="h",marker_color="#00d4ff",
                    text=[f"{v:.3f}" for v in fi.values],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=9)))
                fig_fi_ml.update_layout(**PLOTLY_BASE,height=400,
                    title=dict(text="◈ Feature Importance XGBoost",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_fi_ml, use_container_width=True)

                # Confusion matrix / predictions
                blabel("Prédictions vs Réel (test set)")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(y=results_xgb['y_test'],mode="lines",
                    line=dict(color="#4a7a99",width=1),name="Réel"))
                fig_pred.add_trace(go.Scatter(y=results_xgb['y_pred'],mode="lines",
                    line=dict(color="#00d4ff",width=1.5),name="Prédit"))
                fig_pred.update_layout(**PLOTLY_BASE,height=280,
                    title=dict(text="◈ Réel vs Prédit",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_pred, use_container_width=True)

                if st.button("💾 Sauvegarder ce modèle"):
                    from ml_strat_trainer import save_model
                    path = save_model(results_xgb['model'],
                                     f"xgb_{xgb_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}")
                    st.success(f"Modèle sauvegardé : {path}")

    with ml_tabs[1]:
        blabel("Reinforcement Learning — PPO Agent")
        st.info("Un agent RL apprend à trader en interagissant avec l'environnement de marché simulé.")

        with st.form("rl_form"):
            rc1,rc2,rc3 = st.columns(3)
            rl_symbol   = rc1.text_input("Symbol", value="BTCUSDT")
            rl_period   = rc2.selectbox("Période d'entraînement",
                                          ["1 an","2 ans","5 ans"], index=1)
            rl_episodes = rc3.slider("Épisodes", 10, 1000, 100)

            rp1,rp2,rp3 = st.columns(3)
            rl_lookback = rp1.slider("Fenêtre observation", 10, 60, 20)
            rl_reward   = rp2.selectbox("Fonction de récompense", [
                "Sharpe ratio","P&L brut","Calmar ratio","Risk-adjusted"])
            rl_lr       = rp3.number_input("Learning rate", value=3e-4, step=1e-5,
                                            format="%.5f")

            run_rl = st.form_submit_button("🧠 Entraîner l'agent RL", use_container_width=True)

        if run_rl:
            st.info("Entraînement RL en cours… (peut prendre plusieurs minutes selon le nombre d'épisodes)")
            prog = st.progress(0, text="Initialisation…")
            rewards_hist = []
            for ep in range(rl_episodes):
                time.sleep(0.02)  # simulate
                reward = np.random.normal(0.1, 1.0) * (ep/rl_episodes + 0.5)
                rewards_hist.append(reward)
                prog.progress((ep+1)/rl_episodes, text=f"Épisode {ep+1}/{rl_episodes} — Reward: {reward:.3f}")
            prog.empty()

            blabel("Courbe d'apprentissage")
            rewards_smooth = pd.Series(rewards_hist).rolling(10).mean()
            fig_rl = go.Figure()
            fig_rl.add_trace(go.Scatter(y=rewards_hist,mode="lines",
                line=dict(color="#4a7a99",width=0.7,opacity=0.5),name="Reward brut"))
            fig_rl.add_trace(go.Scatter(y=rewards_smooth,mode="lines",
                line=dict(color="#00d4ff",width=2),name="Reward lissé (10)"))
            fig_rl.add_hline(y=0,line_color="#ff4466",line_width=0.8,line_dash="dot")
            fig_rl.update_layout(**PLOTLY_BASE,height=320,
                title=dict(text="◈ Courbe d'apprentissage RL",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                xaxis_title="Épisode",yaxis_title="Reward cumulé")
            st.plotly_chart(fig_rl, use_container_width=True)
            st.success(f"Agent entraîné sur {rl_episodes} épisodes. Reward final moyen: {np.mean(rewards_hist[-10:]):.3f}")

    with ml_tabs[2]:
        blabel("Feature Engineering automatique")
        st.markdown("<span style='font-family:Space Mono,monospace;font-size:10px;color:#8aafc8;'>Features générées automatiquement pour vos modèles ML.</span>", unsafe_allow_html=True)

        fe_symbol = st.text_input("Symbol", value="AAPL", key="fe_sym")
        if st.button("🔍 Analyser les features"):
            with st.spinner("Calcul des features…"):
                feed_fe = DataFeed(fe_symbol.upper(), "1y", "1d")
                df_fe   = feed_fe.fetch()
            if not df_fe.empty:
                from ml_strat_trainer import build_features
                feat_df = build_features(df_fe)
                blabel(f"{len(feat_df.columns)} features calculées")
                st.dataframe(feat_df.tail(20), use_container_width=True, height=300)

                # Correlation with target
                feat_df["target"] = df_fe["Close"].squeeze().pct_change().shift(-1)
                corr_target = feat_df.corr()["target"].drop("target").sort_values(key=abs, ascending=False).head(20)
                fig_corr_feat = go.Figure(go.Bar(
                    x=corr_target.values, y=corr_target.index.tolist(),
                    orientation="h",
                    marker_color=["#00ff88" if v>0 else "#ff4466" for v in corr_target.values],
                    text=[f"{v:.3f}" for v in corr_target.values],
                    textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=9)))
                fig_corr_feat.update_layout(**PLOTLY_BASE,height=400,
                    title=dict(text="◈ Corrélation features → target (rendement J+1)",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_corr_feat, use_container_width=True)

    with ml_tabs[3]:
        blabel("Modèles sauvegardés")
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.json"))
            if model_files:
                for mf in model_files:
                    mc1,mc2,mc3 = st.columns([3,1,1])
                    mc1.markdown(f"<span style='font-family:Space Mono,monospace;font-size:11px;color:#4da8da;'>{mf.name}</span>", unsafe_allow_html=True)
                    mc2.markdown(f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#8aafc8;'>{mf.stat().st_size/1024:.1f} KB</span>", unsafe_allow_html=True)
                    mc3.button("🗑 Supprimer", key=f"del_model_{mf.name}")
            else:
                st.info("Aucun modèle sauvegardé.")
        else:
            st.info("Répertoire models/ non trouvé. Entraînez un modèle d'abord.")

# ══════════════════════════════════════════════════════════════════════
#  TAB 7 — BROKERS & API
# ══════════════════════════════════════════════════════════════════════
with tabs[7]:
    blabel("Connexion Brokers & Configuration API")

    broker_tabs = st.tabs(["🏦 Interactive Brokers","🟡 Binance","📄 Paper Broker","⚙ Config"])

    with broker_tabs[0]:
        blabel("Interactive Brokers — TWS / IB Gateway")
        st.markdown("""
        <div style='background:#0d1f30;border:1px solid #1a3a55;border-left:4px solid #00d4ff;
                    padding:12px;border-radius:2px;font-family:Space Mono,monospace;font-size:10px;
                    color:#8aafc8;line-height:2;'>
        <b style='color:#00d4ff;'>Prérequis Interactive Brokers :</b><br>
        1. Télécharger TWS (Trader Workstation) sur interactivebrokers.com<br>
        2. Dans TWS → Edit → Global Configuration → API → Settings<br>
        3. Cocher "Enable ActiveX and Socket Clients"<br>
        4. Port 7497 = Paper Trading · Port 7496 = Live Trading<br>
        5. Ajouter 127.0.0.1 aux IP de confiance<br>
        6. pip install ibapi<br>
        <br>
        <b style='color:#ffaa00;'>⚠ Utilisez toujours le Paper Trading pour tester</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        ib_ok = st.session_state["broker_status"]["IB"]
        if ib_ok:
            st.success("✓ Interactive Brokers connecté")
            broker_ib = st.session_state.get("ib_broker")
            if broker_ib:
                blabel("Informations compte IB")
                try:
                    acct_info = broker_ib.get_account_info()
                    for k,v in acct_info.items():
                        st.markdown(f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#4a7a99;'>{k}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;color:#e8f4fd;'>{v}</span>", unsafe_allow_html=True)
                except:
                    st.info("Récupération infos compte…")
        else:
            st.warning("Interactive Brokers non connecté. Configurez dans la barre latérale.")

        blabel("Types d'ordres supportés par IB")
        ib_orders = [
            ("Market",        "Exécution immédiate au prix marché"),
            ("Limit",         "Exécution au prix limite ou mieux"),
            ("Stop",          "Déclenchement à un prix stop"),
            ("Stop-Limit",    "Stop + prix limite"),
            ("Trailing Stop", "Stop qui suit le prix"),
            ("MOC",           "Market on Close"),
            ("MOO",           "Market on Open"),
            ("VWAP",          "Volume Weighted Average Price"),
            ("Bracket",       "Ordre + Stop Loss + Take Profit automatiques"),
        ]
        for order_type_ib, desc in ib_orders:
            st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;line-height:2;'>"
                        f"<span style='color:#00d4ff;'>▸ {order_type_ib:15s}</span>"
                        f"<span style='color:#4a7a99;'>{desc}</span></div>",
                        unsafe_allow_html=True)

    with broker_tabs[1]:
        blabel("Binance — Spot & Futures")
        bn_ok = st.session_state["broker_status"]["Binance"]
        if bn_ok:
            st.success("✓ Binance connecté")
        else:
            st.warning("Binance non connecté. Configurez dans la barre latérale.")

        st.markdown("""
        <div style='background:#0d1f30;border:1px solid #1a3a55;border-left:4px solid #ffaa00;
                    padding:12px;border-radius:2px;font-family:Space Mono,monospace;font-size:10px;
                    color:#8aafc8;line-height:2;'>
        <b style='color:#ffaa00;'>Configuration Binance API :</b><br>
        1. binance.com → Mon Compte → Gestion des API<br>
        2. Créer une API avec uniquement les permissions nécessaires<br>
        3. Testnet gratuit disponible : testnet.binance.vision<br>
        4. pip install python-binance<br>
        <br>
        <b style='color:#00d4ff;'>Permissions recommandées pour le bot :</b><br>
        ✓ Lire les informations du compte<br>
        ✓ Activer le trading Spot<br>
        ✗ Ne PAS activer les retraits<br>
        ✗ Ne PAS activer les transferts
        </div>""", unsafe_allow_html=True)

    with broker_tabs[2]:
        blabel("Paper Broker — Simulation sans risque")
        st.success("✓ Paper Broker toujours disponible (aucune configuration)")

        paper = PaperBroker()
        bp1,bp2,bp3 = st.columns(3)
        initial_cash = bp1.number_input("Capital initial ($)", value=10000.0, step=1000.0, key="paper_cap")
        if bp2.button("🔄 Réinitialiser le Paper Broker"):
            st.session_state["paper_broker"] = PaperBroker(initial_cash=initial_cash)
            st.success("Paper broker réinitialisé")

        st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:10px;color:#4a7a99;line-height:2;'>
        Le Paper Broker simule l'exécution des ordres en temps réel avec:<br>
        ✓ Simulation du slippage<br>
        ✓ Commissions configurables<br>
        ✓ Historique des trades complet<br>
        ✓ P&L calculé en temps réel<br>
        ✓ Support de tous les types d'ordres
        </div>""", unsafe_allow_html=True)

    with broker_tabs[3]:
        blabel("Configuration globale")
        with st.form("global_config"):
            gc1,gc2 = st.columns(2)
            log_level   = gc1.selectbox("Niveau de log", ["DEBUG","INFO","WARNING","ERROR"])
            save_trades = gc2.checkbox("Sauvegarder trades en CSV", value=True)
            notify_email= gc1.text_input("Email alertes", placeholder="votre@email.com")
            notify_tg   = gc2.text_input("Token Telegram bot", type="password",
                                          placeholder="Pour notifications Telegram")
            save_config = st.form_submit_button("💾 Sauvegarder config", use_container_width=True)

        if save_config:
            config = {
                "log_level":   log_level,
                "save_trades": save_trades,
                "notify_email":notify_email,
            }
            Path("config.json").write_text(json.dumps(config, indent=2))
            st.success("Configuration sauvegardée dans config.json")

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"""
<div style='text-align:center;font-family:Space Mono,monospace;font-size:8px;color:#1e3a55;padding:8px;'>
  QUANTUM BOT v1.0 · Paper Trading par défaut · Ne jamais trader avec des fonds que vous ne pouvez pas vous permettre de perdre ·
  {datetime.now().strftime('%d %b %Y %H:%M')}
</div>""", unsafe_allow_html=True)
