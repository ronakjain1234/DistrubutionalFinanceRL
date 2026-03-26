# Risk-Sensitive Distributional Offline RL for Cryptocurrency Trading

This project investigates whether **risk-sensitive distributional offline deep reinforcement learning** can produce more robust trading policies than a simple buy-and-hold benchmark in cryptocurrency markets.

Cryptocurrency markets are extremely volatile and exhibit large tail events. Optimizing only expected returns can lead to unstable, brittle policies. Offline RL is attractive because it learns solely from historical data, but it also introduces its own failure modes.

## The Problem: Extrapolation Error in Offline RL

In **offline reinforcement learning**, the agent learns only from a **fixed historical dataset** and cannot explore new actions. This creates a risk called **extrapolation error**.

The issue is:

1. The dataset contains only **certain state–action pairs** (what traders historically did).
2. The learned value function may estimate **Q-values for actions that rarely or never appeared in the dataset**.
3. Neural networks can produce **overly optimistic estimates for these unseen actions**.
4. The policy then chooses those actions, even though there is **no real data supporting them**.

In cryptocurrency markets this risk is larger because assets like **Bitcoin** and **Ethereum** have **rapidly changing market regimes**. States that occur in 2024–2025 may not closely resemble those in earlier training periods.

So the concern is:

> The learned policy might take actions in states where the dataset provides little evidence about the consequences.

## The Core Solution: Conservative & Uncertainty-Aware Value Estimation

The main defense is **conservative and uncertainty-aware value estimation**.

This project uses **Conservative Q-Learning (CQL)**, which modifies Q-learning to **penalize high value estimates for actions not well supported by the dataset**. This discourages the policy from selecting actions that are **outside the historical data distribution**.

On top of this, we use **distributional RL** (quantile regression) to model the full return distribution instead of only its mean. This allows risk-sensitive decision rules that focus on downside risk or low-quantile performance.

## Additional Protection: Ensemble Uncertainty

To further reduce extrapolation risk, the value function can be estimated using **multiple Q-networks (an ensemble)**.

Each network learns the value of a state–action pair independently. If the dataset **poorly covers that pair**, the networks will disagree. The policy then penalizes actions with high disagreement.

Conceptually:

```text
Estimated value = average Q-value − uncertainty penalty
```

So actions that are **uncertain or poorly supported** automatically become less attractive.

## Why This Makes Sense

These mechanisms work together to keep the policy **close to the data distribution**.

- **Conservative Q-learning** prevents unrealistic value estimates for unseen actions.
- **Distributional RL** exposes the whole return distribution, enabling risk-sensitive criteria.
- **Ensemble uncertainty** detects when the model is unsure because the dataset lacks similar examples.

The policy therefore favors **actions that have strong historical support**. When the model encounters unfamiliar market conditions, it behaves more cautiously rather than exploiting unreliable predictions. This is exactly the behavior needed in highly volatile markets like cryptocurrency trading.

## One-Sentence Summary

The key challenge in offline RL is that the agent may estimate values for state–action pairs that are poorly represented in the dataset; using **Conservative Q-Learning combined with distributional value estimation and ensemble-based uncertainty penalties** discourages the policy from choosing unsupported actions, making the learned trading strategy more robust to distribution shifts in markets such as **Bitcoin** and **Ethereum**.

## Project Goal

- Build an **offline RL** agent (distributional extension of Conservative Q-Learning) that learns from historical BTC data.
- Use **quantile regression** to model the full return distribution and select actions based on downside risk.
- Evaluate **out of sample** and compare to a **buy-and-hold** baseline (annualized return, drawdowns, stability).

## High-Level Plan

We will build the project in stages:

1. **Project scaffolding & dependencies** – set up folders, environment, and base scripts.
2. **Data acquisition & preprocessing (BTC)** – download/clean price data and engineer features.
3. **Portfolio mechanics, rewards, and benchmarks** – define actions, PnL, and buy-and-hold baseline.
4. **Offline trading environment** – a Gym-style env wrapped around the BTC dataset.
5. **Offline replay buffer** – construct a dataset of transitions from a behavior policy.
6. **Baseline agents** – standard RL (e.g., DQN) as a reference.
7. **Conservative Q-Learning (CQL)** – a robust offline RL baseline.
8. **Distributional RL extension** – quantile-based Q-networks.
9. **Risk-sensitive policies** – decision rules based on quantiles / CVaR-style objectives.
10. **Evaluation & visualization** – metrics and plots vs buy-and-hold.
11. **Documentation & reproducibility** – configs, logs, and experiment descriptions.
12. **Extensions** – multi-asset portfolios, higher frequency data, alternative risk measures.

## Project Layout

```text
DistrubutionalFinanceRL/
├── src/
│   ├── data/          # Data download, cleaning, feature engineering
│   ├── env/           # Offline trading environment, portfolio simulation
│   ├── agents/        # RL agents (CQL, distributional, risk-sensitive)
│   └── experiments/   # Training scripts, evaluation, run_experiment
├── data/
│   ├── raw/           # Raw candle data (CSV/Parquet)
│   └── processed/     # Features, splits, offline datasets
├── notebooks/         # Exploratory analysis and debugging
├── configs/           # Experiment configs (YAML/JSON)
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

## Detailed Roadmap (Steps 1–12)

### Step 1: Project scaffolding & dependencies

- **Goal**: Set up a clean Python project with clear structure and required libraries.
- **Tasks**:
  - Create a basic repo layout:
    - `src/` for code (`src/data`, `src/env`, `src/agents`, `src/experiments`).
    - `notebooks/` for exploratory analysis.
    - `data/raw`, `data/processed` folders for market data.
    - `configs/` for experiment configs (YAML/JSON) later.
  - Set up Python environment and add core deps:
    - Data/ML: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`.
    - RL: `torch`, `gymnasium` (or `gym`), `d3rlpy`, optionally `tensorboard`/`wandb`.
    - Utilities: `tqdm`, `pydantic`, optional config tools like `hydra`/`omegaconf`.
  - Add a minimal `run_experiment.py` stub in `src/experiments`.

### Step 2: Data acquisition & preprocessing for BTC

- **Goal**: Build a reproducible pipeline to obtain and preprocess BTC price data into a clean panel suitable for offline RL.
- **Tasks**:
  - Choose data source and frequency (start with **daily BTC-USD candles** from 2017-07 to 2025-12) and standardize to CSV/Parquet in `data/raw`.
  - Implement `src/data/download_btc_data.py` that:
    - Downloads daily candles from Coinbase Exchange (or loads from local CSV/Parquet).
    - Ensures consistent timezone and no duplicate timestamps.
    - Fills small gaps or drops days with missing values according to a clear rule.
  - Implement `src/data/make_features.py` that constructs state features per timestamp:
    - Rolling log returns over different windows (e.g. 1, 5, 20 periods).
    - Realized volatility (rolling std of returns).
    - Momentum indicators (moving averages, RSI, MACD, etc.).
    - Optional volume-based features.
  - Normalize/scale features (e.g. z-score) using training-period statistics only (via `src/data/split_dataset.py`).
  - Split the timeline into train/validation/test (e.g. 2017–2021 train, 2022–2023 val, 2024–2025 test) and store results in `data/processed`.

#### Quickstart: Step 2 (BTC daily)

Run the end-to-end daily pipeline (defaults: BTC-USD, 2017-07-01 to 2025-12-31):

```bash
python -m src.data.download_btc_data
python -m src.data.make_features
python -m src.data.split_dataset
```

Expected outputs:

- `data/raw/btc_daily.parquet`
- `data/raw/btc_daily.csv`
- `data/processed/btc_daily_features.parquet`
- `data/processed/btc_daily_train.parquet`
- `data/processed/btc_daily_val.parquet`
- `data/processed/btc_daily_test.parquet`

### Step 3: Portfolio mechanics, rewards, and benchmarks

- **Goal**: Formalize how actions translate to positions and PnL so environment and agents are well-defined.
- **Tasks**:
  - Specify the **action space** for BTC:
    - Discrete actions: {short, flat, long}, e.g. {-1, 0, +1} sign of exposure with fixed leverage.
  - Implement a simple **portfolio simulator** in `src/env/portfolio.py` that, given price relatives and actions, computes:
    - Position changes, PnL, and account value.
    - Transaction costs and slippage (e.g. proportional fee per turnover).
  - Define the **reward function**:
    - Start with log-return of portfolio after costs per step.
    - Allow for later variants (e.g. downside-penalized returns, Sharpe-like adjustments) without changing the interface.
  - Implement the **buy-and-hold benchmark** in `src/experiments/baselines.py`:
    - Always-long BTC policy that computes equity curve and metrics from the same data splits.

### Step 4: Offline trading environment (Gym-style)

- **Goal**: Implement a reproducible, deterministic offline environment wrapping the BTC dataset.
- **Tasks**:
  - Implement `OfflineTradingEnv` in `src/env/offline_trading_env.py`:
    - Observation at time t: feature vector (and maybe current position).
    - Action: integer in {0,1,2} mapped to short/flat/long in the portfolio module.
    - `step(action)` returns next observation, scalar reward, done flag, and info (e.g. current equity, drawdown).
    - Environment iterates sequentially over the preprocessed dataset (no on-the-fly data fetching).
  - Ensure compatibility with `gymnasium`/`gym` spaces so it can be used with `d3rlpy` or other RL libraries.
  - Add simple smoke tests / a notebook (e.g. `notebooks/env_debug.ipynb`) to step through a few episodes and visually inspect PnL and drawdowns.

### Step 5: Build offline dataset / replay buffer

- **Goal**: Construct an offline dataset of (`state`, `action`, `reward`, `next_state`, `done`) tuples suitable for offline RL.
- **Tasks**:
  - Decide on a **behavior policy** that generates the offline data (e.g. buy-and-hold, random, or simple trend-following heuristic).
  - Implement `build_offline_dataset.py` in `src/data` that:
    - Runs the chosen behavior policy through `OfflineTradingEnv` over the training period.
    - Stores transitions into a dataset / replay buffer format (e.g. `d3rlpy` `MDPDataset`) and saves to `data/processed/offline_dataset_`*.
  - Verify dataset statistics (action distribution, returns, etc.) in a notebook.

### Step 6: Baseline RL agents (non-distributional, non-risk-sensitive)

- **Goal**: Establish baseline performance using standard value-based RL on the offline dataset.
- **Tasks**:
  - Implement or configure a simple **DQN** (or similar) in `src/agents/dqn_baseline.py` (or via `d3rlpy`’s built-in algorithms).
  - Add `train_dqn_baseline.py` in `src/experiments` that:
    - Loads the offline dataset.
    - Trains the baseline agent with a few key hyperparameters configurable via CLI or config file.
    - Periodically evaluates the policy on validation period via environment rollouts.
  - Compare trained policy vs buy-and-hold on validation and test sets using annualized return, volatility, and max drawdown.

### Step 7: Conservative Q-Learning (CQL) offline agent

- **Goal**: Implement or configure a CQL agent as a robust offline RL baseline.
- **Tasks**:
  - Use `d3rlpy`’s CQL implementation (if available) or implement a minimal CQL variant in `src/agents/cql.py`.
  - Add `train_cql.py` in `src/experiments` that:
    - Loads the same offline dataset.
    - Trains CQL with tunable conservatism parameter \alpha and network architecture.
    - Logs training loss, Q-value statistics, and conservatism behavior.
  - Add evaluation code common to all agents in `src/experiments/eval_policies.py`.
  - Compare CQL policy vs DQN baseline vs buy-and-hold.

### Step 8: Distributional RL extension (quantile regression)

- **Goal**: Extend value estimation to model the full return distribution using quantile regression.
- **Tasks**:
  - Implement a distributional Q-network in `src/agents/distributional_qnet.py`:
    - Output N quantiles per action (e.g. 20–51 quantiles).
    - Use quantile Huber loss for training.
  - Integrate quantile outputs into CQL training to obtain a **distributional CQL** variant (or configure `d3rlpy` if it already supports distributional CQL-style algorithms).
  - Add `train_dist_cql.py` in `src/experiments` with hyperparameters:
    - Number of quantiles, discount factor, \alpha for conservatism, learning rate, network size, etc.
  - Log and visualize estimated quantile functions over time/actions for interpretability.

### Step 9: Risk-sensitive policy from learned quantiles

- **Goal**: Use the learned return distribution to define a risk-sensitive policy that emphasizes high-probability favorable outcomes and penalizes downside risk.
- **Tasks**:
  - Define risk-sensitive action selection rules in `src/agents/risk_policies.py`, for example:
    - Maximize a lower quantile (e.g. 10th or 20th percentile of return) instead of mean.
    - Or maximize mean minus \lambda times a downside risk term (e.g. CVaR or lower-tail variance).
  - Implement a small evaluation harness that can plug different action selection functionals into the same trained distributional Q model.
  - Run controlled experiments comparing:
    - Mean-based greedy policy.
    - Lower-quantile-based policy.
    - CVaR-based or other risk-adjusted policies.
  - Record the trade-off between return and drawdown for different risk settings.

### Step 10: Evaluation, metrics, and visualization

- **Goal**: Build a robust evaluation suite and compare all methods to buy-and-hold out of sample.
- **Tasks**:
  - Implement evaluation utilities in `src/experiments/metrics.py` for:
    - Annualized return and annualized volatility.
    - Sharpe ratio, max drawdown, Calmar ratio.
    - Hit ratio (fraction of profitable trades), turnover, and transaction cost impact.
  - Implement plotting utilities in `src/experiments/plots.py` for:
    - Equity curves over time for each policy vs buy-and-hold.
    - Drawdown curves and distribution of period returns.
    - Risk-return scatter plots across hyperparameter sweeps (e.g. different \alpha or risk-sensitive parameters).
  - Create a main script `run_all.py` in `src/experiments` that can reproduce key experiments end-to-end.

### Step 11: Documentation, experiments log, and reproducibility

- **Goal**: Make the project understandable and reproducible for you and others.
- **Tasks**:
  - Expand this `README.md` with more concrete commands and example results as they become available.
  - Add experiment configuration files in `configs/` (e.g. YAML) capturing main training/evaluation settings.
  - Optionally integrate experiment tracking (e.g. `tensorboard` or `wandb`) for loss curves and metrics.

### Step 12: Extensions (optional, later)

- **Multi-asset extension**: extend env and state to handle BTC + ETH (or more), with position vectors and allocation constraints.
- **Higher frequency data**: move from daily to hourly or sub-hourly, with attention to transaction costs and microstructure effects.
- **Alternative risk measures**: experiment with downside deviation, drawdown-sensitive rewards, or utility-based objectives.
- **Modeling choices**: compare different network architectures (e.g. temporal conv nets, transformers) for capturing long-range dependencies.

