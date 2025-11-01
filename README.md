# AI Trading Beast

AI Trading Beast is a multi-market trading framework that combines classic machine learning and optional reinforcement learning to trade crypto (via CCXT exchanges) and forex/metals (via the OANDA API). It includes Telegram-based supervision, ATR-driven risk management, and deployment recipes for Docker and Render.

## Features

- **Multi venue**: Binance/Bybit/KuCoin via CCXT, plus OANDA for FX/Gold.
- **Order support**: Market/limit orders, stop-loss & take-profit scaffolding, reduce-only support, ATR trailing stops.
- **Models**: LightGBM direction classifier and optional Stable-Baselines3 (PPO/DQN) policy.
- **Risk controls**: ATR sizing, percent risk per trade, leverage caps, and daily loss kill switch.
- **Operations**: Telegram bot for control (/start, /pause, /resume, /status, /summary, /risk).
- **Backtesting**: VectorBT signal replay and walk-forward evaluation helpers.
- **Deployment ready**: Dockerfile, docker-compose, and Render deployment checklist.

## Project Structure

```
ai-trading-beast/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ config.yaml
├─ run.py
├─ Dockerfile
├─ docker-compose.yml
├─ core/
│  ├─ engine.py
│  ├─ data.py
│  ├─ features.py
│  ├─ model.py
│  ├─ rl_env.py
│  ├─ rl_train.py
│  ├─ risk.py
│  └─ utils.py
├─ brokers/
│  ├─ ccxt_client.py
│  └─ oanda_client.py
├─ telegram_bot/
│  ├─ bot.py
│  └─ handlers.py
└─ backtesting/
   ├─ backtest_vectorbt.py
   └─ walkforward.py
```

## Quick Start

1. **Clone & install**
   ```bash
   git clone https://github.com/your-org/ai-trading-beast.git
   cd ai-trading-beast
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Environment variables**
   ```bash
   cp .env.example .env
   # Populate Telegram + exchange credentials
   ```
3. **Configuration**
   Update `config.yaml`:
   - `general.base_timeframe` (e.g. `1h`, `15m`)
   - `ml.model_type`: `lightgbm` (default) or `rl`
   - `risk` block for ATR sizing and loss caps
   - `markets` list for symbols (BTC/USDT, ETH/USDT, XAU_USD)
4. **Run in paper mode**
   ```bash
   python run.py
   ```
   The Telegram bot will come online and start polling for signals.

## Configuration Overview

- **`.env`** holds secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `CCXT_API_KEY`, `CCXT_API_SECRET`, optional `CCXT_PASSWORD`, and OANDA `OANDA_ACCOUNT_ID`, `OANDA_API_TOKEN`, `OANDA_ENV` (practice/live).
- **`config.yaml`** controls trading behaviour:
  - `general.prediction_horizon_bars`: future bar count for label creation.
  - `risk`: `risk_per_trade_pct`, `max_daily_loss_pct`, `sl_atr_mult`, `tp_rr`, `trailing_atr_mult`, `leverage_cap`.
  - `ml`: `use_model`, `model_type`, probability threshold, LightGBM model path.
  - `rl`: toggle RL agent, algorithm choice, env window, reward, and cost assumptions.
  - `execution`: market order allowance, reduce-only flags, slippage assumptions, heartbeat.

## Classic ML Workflow (LightGBM)

1. **Fetch & feature**
   ```python
   from core.data import load_time_series
   from core.features import generate_features

   df = load_time_series("BTC/USDT", "1h")
   features = generate_features(df)
   ```
2. **Train classifier**
   ```python
   from core.model import DirectionClassifier

   clf = DirectionClassifier(probability_threshold=0.6)
   result = clf.fit(features, horizon=3)
   print(result.metrics)
   ```
   The trained model is saved to `models/lightgbm_default.txt`. Update `config.yaml` if you use a custom path.
3. **Live inference**
   - Ensure `ml.model_type: lightgbm` and `ml.model_path` points to your model.
   - Signals require both the probability threshold and EMA trend filter (`EMA12 > EMA26`).

## Reinforcement Learning Mode

1. **Prepare data**
   Ensure target symbol exists in `config.yaml` with historical data accessible.
2. **Train policy**
   ```bash
   python -m core.rl_train BTC/USDT
   ```
   (Call `train_rl("BTC/USDT")` from a Python shell for programmatic control.)
3. **Configuration**
   - Set `ml.model_type: rl` and `rl.model_path` to the generated checkpoint (`models/rl/ppo/BTC_USDT_YYYYMMDD-HHMMSS.zip`).
   - Adjust `rl.algo`, `rl.total_timesteps`, `rl.env_window`, and reward shaping as needed.
4. **Live inference**
   The engine loads the Stable-Baselines3 policy and maps discrete actions to orders per bar while respecting risk guardrails.

## Backtesting & Walk-Forward

- **VectorBT replay**
  ```bash
  python backtesting/backtest_vectorbt.py
  ```
  Generates per-symbol statistics with identical fee/slippage assumptions.
- **Walk-forward evaluation**
  ```python
  from backtesting.walkforward import walk_forward
  results = walk_forward(features, horizon=3, train_size=1000, test_size=200)
  for window in results:
      print(window)
  ```

## Telegram Bot

- Create a bot via **BotFather** and obtain the token.
- Find your chat ID via [@userinfobot](https://t.me/userinfobot) or start the bot and inspect updates.
- Supported commands:
  - `/start` — acknowledge bot is online
  - `/pause` — halt trading loop
  - `/resume` — resume processing
  - `/status` — engine status and open positions
  - `/summary` — placeholder for PnL summary
  - `/risk` — display current risk settings

All outbound messages and logs are prefixed with “AI Trading Beast” for easy filtering.

## Docker Usage

### Build & run locally
```bash
docker build -t ai-trading-beast .
docker run --env-file .env ai-trading-beast
```

### docker-compose
```bash
docker-compose up --build
```
Compose mounts the repository for live reloads, shares `.env`, and includes a healthcheck.

## Deploying on Render

1. **Create a new Web Service** pointing to your repository.
2. **Environment**
   - Runtime: Docker
   - Build Command: `docker build -t ai-trading-beast .`
   - Start Command: `python run.py`
3. **Environment variables**
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - EXCHANGE (e.g. `binance`)
   - CCXT_API_KEY / CCXT_API_SECRET / CCXT_PASSWORD
   - OANDA_ACCOUNT_ID / OANDA_API_TOKEN / OANDA_ENV
4. **Scaling**
   - Use a background worker or web service with at least 512MB RAM.
   - Enable automatic restarts on failure.
5. **Monitoring**
   - Render dashboard exposes logs (`AI Trading Beast ...`).
   - Configure healthcheck pings via Render settings.

## Safety Checklist

- Use **testnet/practice** accounts first (Binance testnet, OANDA practice).
- Apply **principle of least privilege** API keys (read/trade only, no withdrawal).
- Respect exchange **rate limits** (CCXT rate limiting is enabled by default).
- Validate configuration on paper mode before enabling live mode.
- Monitor Telegram alerts for order fills, errors, and kill-switch triggers.
- Keep dependencies up to date and pin versions as provided in `requirements.txt`.

## Contributing

Issues and PRs are welcome. Please lint and run your backtests before submitting.

## License

MIT License © 2024 AI Trading Beast contributors.
