"""Optimizers for neural network training (CPU implementation)."""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, learning_rate: float = 0.001):
        """Initialize optimizer.
        
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate
    
    def update(self, params: Dict, grads: Dict):
        """Update parameters using gradients.
        
        Args:
            params: Dictionary of parameters (weights, biases)
            grads: Dictionary of gradients
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.001):
        """Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)
    
    def update(self, params: Dict, grads: Dict):
        """Update parameters using SGD.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for key in params:
            if key in grads:
                params[key] -= self.learning_rate * grads[key]


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small epsilon for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Per-parameter moment estimates
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def update(self, params: Dict, grads: Dict):
        """Update parameters using Adam.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        self.t += 1
        
        for key in params:
            if key not in grads:
                continue
            
            param = params[key]
            grad = grads[key]
            
            # Initialize moments if needed
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        """Reset optimizer state."""
        self.m.clear()
        self.v.clear()
        self.t = 0


