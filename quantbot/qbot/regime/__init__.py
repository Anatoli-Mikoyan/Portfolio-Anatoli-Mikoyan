"""Couche de détection de régime (§7). Le clustering et le HMM requièrent scikit-learn / hmmlearn."""
from .base import RegimeDetector, RegimeSeries, LookaheadError
from .features import build_regime_matrix, looks_rolling_normalized
from .detectors import (
    RuleBasedDetector, ClusteringDetector, HMMDetector, build_detector,
)
from .evaluate import (
    RegimeUsefulness, conditional_performance, regime_usefulness, compare_detectors,
    strategy_regime_map, lookahead_gain,
)

__all__ = [
    "RegimeDetector", "RegimeSeries", "LookaheadError",
    "build_regime_matrix", "looks_rolling_normalized",
    "RuleBasedDetector", "ClusteringDetector", "HMMDetector", "build_detector",
    "RegimeUsefulness", "conditional_performance", "regime_usefulness",
    "compare_detectors", "strategy_regime_map", "lookahead_gain",
]
