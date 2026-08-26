"""Fixtures partagées."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.config import CostConfig, EnvConfig, FeatureConfig
from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv

BPY = 6240.0


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return generate_synthetic_ohlcv(n=6000, seed=42).drop(columns=["regime"])


@pytest.fixture(scope="session")
def ohlcv_with_signal() -> pd.DataFrame:
    market = RegimeSwitchingGBM(autocorr=0.25, persistence=0.997)
    return generate_synthetic_ohlcv(n=8000, seed=7, model=market).drop(columns=["regime"])


@pytest.fixture(scope="session")
def feature_cfg() -> FeatureConfig:
    return FeatureConfig(
        returns_windows=(1, 5, 20), vol_windows=(10, 20), ema_windows=(10, 30),
        use_microstructure=True, use_calendar=True, scaler_window=300,
    )


@pytest.fixture
def zero_cost() -> CostConfig:
    return CostConfig(spread_bps=0.0, commission_bps=0.0, slippage_model="none",
                      financing_bps_per_bar=0.0, min_trade_size=0.0)


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(window=16, positions=(-1.0, 0.0, 1.0), vol_target=None,
                     episode_length=None, random_start=False, max_drawdown_stop=None)
