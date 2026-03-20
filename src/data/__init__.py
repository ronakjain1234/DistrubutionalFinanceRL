"""
Data acquisition and feature engineering for the BTC offline RL experiments.

Step 2 in the README corresponds to:
1) Download raw BTC daily OHLCV data (see `download_btc_data.py`)
2) Build a feature panel per timestamp (see `make_features.py`)
3) Split by time and normalize using training-period statistics only (see `split_dataset.py`)
"""

from __future__ import annotations

__all__ = [
    "download_btc_data",
    "make_features",
    "split_dataset",
]

