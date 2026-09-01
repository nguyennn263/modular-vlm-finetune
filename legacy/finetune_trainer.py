"""
Backward compatibility wrapper.
Import from trainer.py instead.
"""

from .trainer import BridgeTrainer, BridgeFineTuner, TrainConfig

__all__ = ['BridgeTrainer', 'BridgeFineTuner', 'TrainConfig']
