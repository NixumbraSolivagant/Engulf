"""Utility functions for RL module."""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import math


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def normalize_to_range(value: float, min_val: float, max_val: float, 
                       target_min: float = -1.0, target_max: float = 1.0) -> float:
    """Normalize value to target range.
    
    Args:
        value: Value to normalize
        min_val: Source minimum
        max_val: Source maximum
        target_min: Target minimum
        target_max: Target maximum
        
    Returns:
        Normalized value in [target_min, target_max]
    """
    if max_val == min_val:
        return (target_min + target_max) / 2.0
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def clip_action(action: np.ndarray, action_space: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Clip action to valid range.
    
    Args:
        action: Action array
        action_space: Dictionary mapping action names to (min, max) tuples
        
    Returns:
        Clipped action array
    """
    clipped = action.copy()
    for i, (name, (min_val, max_val)) in enumerate(action_space.items()):
        clipped[i] = np.clip(clipped[i], min_val, max_val)
    return clipped


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float = 0.99, lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Array of rewards (length N)
        values: Array of state values (length N+1, includes next_state value)
        dones: Array of done flags (length N)
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns) both of length N
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    
    # values should have length n+1 (current values + next value)
    # Use only first n values for computation
    current_values = values[:n]
    
    for t in reversed(range(n)):
        if dones[t]:
            last_gae = 0.0
        # values[t+1] is the next state value
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - current_values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * last_gae
    
    returns = advantages + current_values
    return advantages, returns


def discount_rewards(rewards: np.ndarray, gamma: float = 0.99, 
                    dones: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute discounted rewards.
    
    Args:
        rewards: Array of rewards
        gamma: Discount factor
        dones: Optional array of done flags
        
    Returns:
        Discounted rewards
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0
    
    for t in reversed(range(len(rewards))):
        if dones is not None and dones[t]:
            running_sum = 0
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    
    return discounted


def flatten_state(state: Dict[str, Any]) -> np.ndarray:
    """Flatten nested state dictionary to array.
    
    Args:
        state: Nested state dictionary
        
    Returns:
        Flattened state array
    """
    result = []
    
    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                _flatten(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for item in obj:
                _flatten(item)
        elif isinstance(obj, (int, float, np.number)):
            result.append(float(obj))
    
    _flatten(state)
    return np.array(result, dtype=np.float32)


def safe_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalize array to unit vector.
    
    Args:
        array: Input array
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Normalized array
    """
    norm = np.linalg.norm(array)
    if norm < eps:
        return np.zeros_like(array)
    return array / norm


def log_sum_exp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-sum-exp in numerically stable way.
    
    Args:
        x: Input array
        axis: Axis along which to compute
        
    Returns:
        Log-sum-exp result
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature.
    
    Args:
        x: Input array
        axis: Axis along which to compute
        temperature: Temperature parameter (higher = more uniform)
        
    Returns:
        Softmax probabilities
    """
    x = x / temperature
    x = x - np.max(x, axis=axis, keepdims=True)  # Numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def exponential_moving_average(current: float, new_value: float, alpha: float = 0.1) -> float:
    """Compute exponential moving average.
    
    Args:
        current: Current EMA value
        new_value: New value to incorporate
        alpha: Smoothing factor (0-1)
        
    Returns:
        Updated EMA value
    """
    return alpha * new_value + (1 - alpha) * current


class RunningStats:
    """Running statistics for state normalization."""
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        """Initialize running statistics.
        
        Args:
            shape: Shape of the state
            epsilon: Small epsilon for numerical stability
        """
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with new sample.
        
        Args:
            x: New sample
        """
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.var = self.var + (x - old_mean) * (x - self.mean)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics.
        
        Args:
            x: Input to normalize
            
        Returns:
            Normalized input
        """
        if self.n < 2:
            return x
        std = np.sqrt(self.var / (self.n - 1)) + self.epsilon
        return (x - self.mean) / std

