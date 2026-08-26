"""Agents RL. Ce sous-paquet est le seul à dépendre de PyTorch."""
from .networks import QNetwork, NoisyLinear, MLPEncoder, GRUEncoder, TCNEncoder, CausalConv1d
from .replay import (
    PrioritizedReplayBuffer, SumTree, NStepAccumulator, Transition, ObsReconstructor,
)
from .rainbow import RainbowAgent, resolve_device
from .trainer import Trainer, evaluate, EvalResult, TrainHistory
from .ensemble import EnsembleAgent

__all__ = [
    "QNetwork", "NoisyLinear", "MLPEncoder", "GRUEncoder", "TCNEncoder", "CausalConv1d",
    "PrioritizedReplayBuffer", "SumTree", "NStepAccumulator", "Transition", "ObsReconstructor",
    "RainbowAgent", "resolve_device",
    "Trainer", "evaluate", "EvalResult", "TrainHistory", "EnsembleAgent",
]
