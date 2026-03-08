# Risk-Sensitive Distributional Offline RL for Cryptocurrency Trading

This project investigates whether **risk-sensitive distributional offline deep reinforcement learning** can produce more robust trading policies than a simple buy-and-hold benchmark in cryptocurrency markets.

## Goal

- Build an **offline RL** agent (distributional extension of Conservative Q-Learning) that learns from historical BTC data.
- Use **quantile regression** to model the full return distribution and select actions based on downside risk.
- Evaluate **out of sample** and compare to a **buy-and-hold** baseline (annualized return, drawdowns, stability).

## Project layout

```
DistrubutionalFinanceRL/
├── src/
│   ├── data/          # Data download, cleaning, feature engineering
│   ├── env/            # Offline trading environment, portfolio simulation
│   ├── agents/         # RL agents (CQL, distributional, risk-sensitive)
│   └── experiments/   # Training scripts, evaluation, run_experiment
├── data/
│   ├── raw/            # Raw candle data (CSV/Parquet)
│   └── processed/      # Features, splits, offline datasets
├── notebooks/          # Exploratory analysis and debugging
├── configs/            # Experiment configs (YAML/JSON)
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. From the project root, run the experiment stub:

   ```bash
   python -m src.experiments.run_experiment
   ```

## Next steps

After scaffolding (Step 1), the plan is: **data pipeline** (BTC download + features) → **trading env + buy-and-hold** → **offline dataset + baselines** → **distributional CQL + risk-sensitive policies** → **evaluation and plots**.
