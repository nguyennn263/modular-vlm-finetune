"""Training: frozen Vintern-1B + trainable bridge."""

from .setup import create_finetune_model, VisionLanguageBridge
from .trainer import BridgeTrainer, BridgeFineTuner, TrainConfig

__all__ = [
    "create_finetune_model",
    "VisionLanguageBridge",
    "BridgeTrainer",
    "BridgeFineTuner",
    "TrainConfig",
]
