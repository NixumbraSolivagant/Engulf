"""Reinforcement Learning module for creatures.

This module implements a full RL system where all creatures use neural networks
for decision-making, with genes providing initial biases and physical constraints.
CPU-optimized, no GPU required.
"""

from .network import PolicyNetwork, ValueNetwork
from .ppo import PPOAgent, PPOConfig
from .state_encoder import StateEncoder
from .action_executor import ActionExecutor
from .reward_shaping import PersonalizedReward, RewardShaping
from .experience_buffer import ExperienceBuffer, PrioritizedBuffer
from .global_manager import GlobalRLManager
from .rl_creature import RLCreature
from .optimizer import Adam, SGD, Optimizer
from .visualizer import RLTrainingVisualizer, create_simple_visualizer

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "PPOAgent",
    "PPOConfig",
    "StateEncoder",
    "ActionExecutor",
    "PersonalizedReward",
    "RewardShaping",
    "ExperienceBuffer",
    "PrioritizedBuffer",
    "GlobalRLManager",
    "RLCreature",
    "Adam",
    "SGD",
    "Optimizer",
    "RLTrainingVisualizer",
    "create_simple_visualizer",
]

