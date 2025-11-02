"""Neural network implementations for RL (CPU-optimized, no GPU).

Lightweight networks using NumPy for CPU efficiency.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional, List
import json
import os


class Layer:
    """Base neural network layer with backpropagation support."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: Optional[str] = None, use_bias: bool = True):
        """Initialize layer.
        
        Args:
            input_size: Input dimension
            output_size: Output dimension
            activation: Activation function name ('relu', 'tanh', 'sigmoid', 'linear')
            use_bias: Whether to use bias
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights (Xavier/Glorot initialization)
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        if use_bias:
            self.bias = np.zeros(output_size, dtype=np.float32)
        else:
            self.bias = None
        
        # Activation function
        self.activation_name = activation or 'linear'
        self.activation = self._get_activation(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
        
        # For backpropagation
        self.last_input = None
        self.last_output = None
        self.last_pre_activation = None  # Before activation
    
    def _get_activation(self, name: Optional[str]):
        """Get activation function."""
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return np.tanh
        elif name == 'sigmoid':
            return lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif name == 'linear' or name is None:
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _get_activation_derivative(self, name: Optional[str]):
        """Get activation derivative for backpropagation."""
        if name == 'relu':
            return lambda x: (x > 0).astype(np.float32)
        elif name == 'tanh':
            return lambda x: 1.0 - np.tanh(x) ** 2
        elif name == 'sigmoid':
            return lambda x: x * (1.0 - x)  # Assumes x is already sigmoid output
        elif name == 'linear' or name is None:
            return lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input array (batch_size, input_size)
            
        Returns:
            Output array (batch_size, output_size)
        """
        self.last_input = x.copy()
        
        # Linear transformation
        output = np.dot(x, self.weights)
        if self.use_bias:
            output += self.bias
        
        # Store pre-activation for backprop
        self.last_pre_activation = output.copy()
        
        # Apply activation
        self.last_output = self.activation(output)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backward pass (backpropagation).
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output (batch_size, output_size)
            
        Returns:
            Tuple of (grad_input, grads_dict)
            - grad_input: Gradient w.r.t. input (batch_size, input_size)
            - grads_dict: Dictionary with 'weights' and 'bias' gradients
        """
        if self.last_input is None or self.last_pre_activation is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        # Gradient through activation
        act_grad = self.activation_derivative(self.last_pre_activation)
        grad_pre_activation = grad_output * act_grad
        
        # Gradient w.r.t. weights: dL/dW = X^T * dL/dY
        grad_weights = np.dot(self.last_input.T, grad_pre_activation)
        
        # Gradient w.r.t. bias: sum over batch
        grad_bias = np.sum(grad_pre_activation, axis=0) if self.use_bias else None
        
        # Gradient w.r.t. input: dL/dX = dL/dY * W^T
        grad_input = np.dot(grad_pre_activation, self.weights.T)
        
        grads = {'weights': grad_weights}
        if grad_bias is not None:
            grads['bias'] = grad_bias
        
        return grad_input, grads
    
    def get_params(self) -> Dict:
        """Get layer parameters."""
        return {
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist() if self.bias is not None else None,
            'activation': self.activation_name,
        }
    
    def set_params(self, params: Dict):
        """Set layer parameters."""
        self.weights = np.array(params['weights'], dtype=np.float32)
        if params['bias'] is not None:
            self.bias = np.array(params['bias'], dtype=np.float32)
        self.activation = self._get_activation(params['activation'])


class PolicyNetwork:
    """Policy network for actor-critic methods (CPU implementation).
    
    Uses NumPy for computation, optimized for CPU without GPU dependencies.
    """
    
    def __init__(self, state_dim: int = 140, action_dim: int = 12,
                 hidden_sizes: Tuple[int, ...] = (128, 64, 32),
                 activation: str = 'tanh'):
        """Initialize policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (continuous)
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        input_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(Layer(input_size, hidden_size, activation))
            input_size = hidden_size
        
        # Output layer: mean and log_std for Gaussian policy
        self.mean_layer = Layer(input_size, action_dim, 'linear')
        self.log_std_layer = Layer(input_size, action_dim, 'linear')
        
        # Initialize log_std to small values (encourage exploration initially)
        self.log_std_layer.weights.fill(0.0)
        self.log_std_layer.bias.fill(-0.5)  # log_std ≈ -0.5 → std ≈ 0.6
        
        self.layers = layers
    
    def forward(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through network.
        
        Args:
            state: State array (batch_size, state_dim) or (state_dim,)
            deterministic: If True, return mean without sampling
            
        Returns:
            Tuple of (action, log_prob) or (mean, log_prob)
        """
        # Handle single sample
        single_sample = len(state.shape) == 1
        if single_sample:
            state = state.reshape(1, -1)
        
        # Forward through hidden layers
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        
        # Get mean and log_std
        mean = self.mean_layer.forward(x)
        log_std = self.log_std_layer.forward(x)
        # Allow higher initial exploration, but will be clamped in _compute_entropy during training
        log_std = np.clip(log_std, -20, 2)  # Clamp log_std for safety
        std = np.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            # Sample from Gaussian
            noise = np.random.normal(0, 1, mean.shape)
            action = mean + std * noise
        
        # Compute log probability
        log_prob = self._gaussian_log_prob(action, mean, log_std)
        
        # Squash action to [-1, 1] using tanh for bounded actions
        action = np.tanh(action)
        
        if single_sample:
            action = action[0]
            log_prob = log_prob[0]
        
        return action, log_prob
    
    def _gaussian_log_prob(self, action: np.ndarray, mean: np.ndarray, 
                          log_std: np.ndarray) -> np.ndarray:
        """Compute log probability of Gaussian distribution."""
        std = np.exp(log_std)
        var = std ** 2
        
        # Log prob before tanh
        pre_tanh_action = np.arctanh(np.clip(action, -0.999, 0.999))
        log_prob = -0.5 * (((pre_tanh_action - mean) / std) ** 2 + 
                          2 * log_std + np.log(2 * np.pi))
        
        # Tanh correction (derivative) - fix numerical stability
        action_sq = np.clip(action ** 2, -1.0 + 1e-6, 1.0 - 1e-6)
        tanh_correction = np.log(1 - action_sq + 1e-8)
        log_prob -= tanh_correction
        
        return np.sum(log_prob, axis=-1)
    
    def get_params(self) -> Dict:
        """Get all network parameters."""
        return {
            'layers': [layer.get_params() for layer in self.layers],
            'mean_layer': self.mean_layer.get_params(),
            'log_std_layer': self.log_std_layer.get_params(),
        }
    
    def set_params(self, params: Dict):
        """Set all network parameters."""
        for i, layer_params in enumerate(params['layers']):
            self.layers[i].set_params(layer_params)
        self.mean_layer.set_params(params['mean_layer'])
        self.log_std_layer.set_params(params['log_std_layer'])
    
    def apply_gene_bias(self, genome_dict: Dict[str, float], bias_scale: float = 0.1):
        """Apply gene-based bias to network (modify output layer biases).
        
        Args:
            genome_dict: Dictionary of gene values
            bias_scale: Scale factor for bias application
        """
        # Extract personality genes
        curiosity = genome_dict.get('curiosity', 0.5)
        aggression = genome_dict.get('aggression', 0.5)
        caution = genome_dict.get('caution', 0.5)
        exploration_drive = genome_dict.get('exploration_drive', 0.5)
        
        # Create bias vector based on genes
        # High curiosity → more exploration
        # High aggression → more speed
        # High caution → more safety
        bias = np.array([
            curiosity * 0.3,  # direction x
            curiosity * 0.3,  # direction y
            aggression * 0.2,  # speed
            exploration_drive * 0.2,  # angular velocity
            exploration_drive * 0.3,  # exploration weight
            curiosity * 0.2,  # resource weight
            caution * 0.3,  # safety weight
            genome_dict.get('sociability', 0.5) * 0.2,  # social weight
            genome_dict.get('risk_tolerance', 0.5) * 0.2,  # patience
            genome_dict.get('risk_tolerance', 0.5) * 0.2,  # risk taking
            genome_dict.get('competitiveness', 0.5) * 0.1,  # competitiveness
            genome_dict.get('adaptation_rate', 0.5) * 0.1,  # adaptation
        ], dtype=np.float32)
        
        # Apply bias to mean layer
        self.mean_layer.bias += bias * bias_scale


class ValueNetwork:
    """Value network for state value estimation."""
    
    def __init__(self, state_dim: int = 140,
                 hidden_sizes: Tuple[int, ...] = (128, 64, 32),
                 activation: str = 'tanh'):
        """Initialize value network.
        
        Args:
            state_dim: State dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        self.state_dim = state_dim
        
        # Build network
        layers = []
        input_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(Layer(input_size, hidden_size, activation))
            input_size = hidden_size
        
        # Output layer: single value
        self.value_layer = Layer(input_size, 1, 'linear')
        self.layers = layers
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            state: State array (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Value array (batch_size,) or scalar
        """
        # Handle single sample
        single_sample = len(state.shape) == 1
        if single_sample:
            state = state.reshape(1, -1)
        
        # Forward through hidden layers
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        
        # Get value
        value = self.value_layer.forward(x)
        
        if single_sample:
            value = value[0, 0]
        else:
            value = value[:, 0]
        
        return value
    
    def get_params(self) -> Dict:
        """Get all network parameters."""
        return {
            'layers': [layer.get_params() for layer in self.layers],
            'value_layer': self.value_layer.get_params(),
        }
    
    def set_params(self, params: Dict):
        """Set all network parameters."""
        for i, layer_params in enumerate(params['layers']):
            self.layers[i].set_params(layer_params)
        self.value_layer.set_params(params['value_layer'])
    
    def save(self, filepath: str):
        """Save network to file."""
        params = self.get_params()
        with open(filepath, 'w') as f:
            json.dump(params, f)
    
    def load(self, filepath: str):
        """Load network from file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.set_params(params)

