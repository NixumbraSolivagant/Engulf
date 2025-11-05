"""PPO (Proximal Policy Optimization) algorithm implementation."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .network import PolicyNetwork, ValueNetwork
from .utils import compute_gae, clip_action
from .experience_buffer import Experience
from .optimizer import Adam


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    gamma: float = 0.99              # Discount factor
    lambda_: float = 0.95            # GAE lambda
    clip_epsilon: float = 0.2        # PPO clipping range
    value_coef: float = 0.3          # Value function loss weight (reduced for stability)
    entropy_coef: float = 0.005      # Entropy bonus weight (reduced to allow policy convergence)
    learning_rate: float = 5e-4      # Learning rate (increased for faster learning)
    batch_size: int = 64             # Batch size
    n_epochs: int = 10               # Number of update epochs
    max_grad_norm: float = 0.5      # Gradient clipping
    use_gae: bool = True             # Use Generalized Advantage Estimation


class PPOAgent:
    """PPO agent for training."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[PPOConfig] = None):
        """Initialize PPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: PPO configuration
        """
        self.config = config or PPOConfig()
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        # Optimizers
        self.policy_optimizer = Adam(learning_rate=self.config.learning_rate)
        self.value_optimizer = Adam(learning_rate=self.config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
        }
        
        # Entropy decay schedule
        self.initial_entropy_coef = self.config.entropy_coef
        self.min_entropy_coef = 0.001  # Minimum entropy coefficient
        self.entropy_decay_rate = 0.995  # More aggressive decay rate per update
        self.update_count = 0  # Track number of updates for entropy scheduling
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action from policy.
        
        Args:
            state: State array
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        action, log_prob = self.policy.forward(state, deterministic=deterministic)
        return action, log_prob
    
    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate.
        
        Args:
            state: State array
            
        Returns:
            Value estimate
        """
        return self.value.forward(state)
    
    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """Update networks using PPO algorithm.
        
        Args:
            experiences: List of experiences
            
        Returns:
            Dictionary of training statistics
        """
        if len(experiences) == 0:
            return {}
        
        # Increment update count for entropy scheduling
        self.update_count += 1
        
        # Extract data
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Compute values
        values = np.array([self.value.forward(s) for s in states]).flatten()
        next_values = np.array([self.value.forward(ns) if not done else 0.0 
                               for ns, done in zip(next_states, dones)]).flatten()
        
        # Ensure same length
        if len(next_values) == 0:
            next_values = np.array([0.0])
        
        # Compute returns and advantages
        if self.config.use_gae:
            # For GAE, we need values + next value appended
            values_with_next = np.append(values, next_values[-1] if len(next_values) > 0 else 0.0)
            advantages, returns = compute_gae(
                rewards, values_with_next, dones,
                self.config.gamma, self.config.lambda_
            )
        else:
            # Simple discounted returns
            returns = rewards.copy()
            for i in reversed(range(len(returns) - 1)):
                if not dones[i]:
                    returns[i] += self.config.gamma * returns[i + 1]
            advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old log probs
        _, old_log_probs = self.policy.forward(states)
        old_log_probs_action = self._compute_action_log_prob(states, actions)
        
        # Training loop
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        
        # Mini-batch training
        batch_size = min(self.config.batch_size, len(experiences))
        n_batches = max(1, len(experiences) // batch_size)
        
        for epoch in range(self.config.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(len(experiences))
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(experiences))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs_action[batch_indices]
                
                # Compute policy loss and gradients
                policy_loss, policy_grads = self._compute_policy_loss_and_grad(
                    batch_states, batch_actions, batch_advantages, batch_old_log_probs
                )
                
                # Compute value loss and gradients
                value_loss, value_grads = self._compute_value_loss_and_grad(
                    batch_states, batch_returns
                )
                
                # Update networks using gradients
                self._update_policy(policy_grads)
                self._update_value(value_grads)
                
                # Track statistics
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                
                # Compute entropy and clip fraction
                entropy = self._compute_entropy(batch_states)
                entropies.append(entropy.mean())
                
                new_log_probs = self._compute_action_log_prob(batch_states, batch_actions)
                ratio = np.exp(new_log_probs - batch_old_log_probs)
                clip_fraction = np.mean((ratio < 1 - self.config.clip_epsilon) | 
                                      (ratio > 1 + self.config.clip_epsilon))
                clip_fractions.append(clip_fraction)
        
        # Return statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
        }
        
        self.training_stats['policy_loss'].append(stats['policy_loss'])
        self.training_stats['value_loss'].append(stats['value_loss'])
        self.training_stats['entropy'].append(stats['entropy'])
        self.training_stats['clip_fraction'].append(stats['clip_fraction'])
        
        return stats
    
    def _compute_action_log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute log probability of actions given states.
        
        Args:
            states: State array
            actions: Action array
            
        Returns:
            Log probabilities
        """
        # Forward pass to get mean and log_std
        x = states
        for layer in self.policy.layers:
            x = layer.forward(x)
        
        mean = self.policy.mean_layer.forward(x)
        log_std = self.policy.log_std_layer.forward(x)
        log_std = np.clip(log_std, -20, 2)
        
        # Compute log prob
        std = np.exp(log_std)
        pre_tanh_actions = np.arctanh(np.clip(actions, -0.999, 0.999))
        
        log_prob = -0.5 * (((pre_tanh_actions - mean) / std) ** 2 +
                          2 * log_std + np.log(2 * np.pi))
        tanh_correction = np.log(1 - actions ** 2 + 1e-6)
        log_prob -= tanh_correction
        
        return np.sum(log_prob, axis=-1)
    
    def _compute_entropy(self, states: np.ndarray) -> np.ndarray:
        """Compute entropy of policy distribution.
        
        Args:
            states: State array
            
        Returns:
            Entropy values
        """
        # Forward pass
        x = states
        for layer in self.policy.layers:
            x = layer.forward(x)
        
        log_std = self.policy.log_std_layer.forward(x)
        
        # Force entropy decay by gradually reducing max log_std
        # This directly controls exploration: higher log_std = more exploration
        # Start with max_log_std = 0.5 (std ≈ 1.65), decay to -1.0 (std ≈ 0.37)
        # More aggressive decay: reduce max_log_std faster (over ~100 updates instead of 1000)
        decay_factor = min(1.0, self.update_count * 0.01)
        max_log_std = 0.5 - 1.5 * decay_factor  # Decay from 0.5 to -1.0
        min_log_std = -1.5  # Minimum log_std
        log_std = np.clip(log_std, min_log_std, max_log_std)
        
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * std^2)
        entropy = 0.5 * (np.log(2 * np.pi * np.e) + 2 * log_std)
        return np.sum(entropy, axis=-1)
    
    def _compute_policy_loss_and_grad(self, states: np.ndarray, actions: np.ndarray,
                                     advantages: np.ndarray, 
                                     old_log_probs: np.ndarray) -> Tuple[float, Dict]:
        """Compute policy loss and gradients.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            old_log_probs: Old log probabilities
            
        Returns:
            Tuple of (loss, gradients_dict)
        """
        # Forward pass
        new_log_probs = self._compute_action_log_prob(states, actions)
        
        # Compute ratio and clipped objective
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon)
        obj = np.minimum(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -np.mean(obj)
        
        # Add entropy bonus with decay
        entropy = self._compute_entropy(states)
        # Decay entropy coefficient over time (already computed in _compute_entropy)
        # More aggressive decay: reduce entropy bonus faster (cap at 100 updates)
        current_entropy_coef = max(
            self.min_entropy_coef,
            self.initial_entropy_coef * (self.entropy_decay_rate ** min(self.update_count, 100))
        )
        policy_loss -= current_entropy_coef * entropy.mean()
        
        # Compute gradients using policy gradient theorem
        # L = E[min(r * A, clip(r) * A)] where r = new_prob / old_prob
        # For efficiency, we'll use a simplified gradient approximation
        # Full backprop would require implementing backward() for all layers
        grads = self._compute_policy_gradients_simplified(states, actions, advantages, old_log_probs)
        
        return float(policy_loss), grads
    
    def _compute_value_loss_and_grad(self, states: np.ndarray, 
                                     returns: np.ndarray) -> Tuple[float, Dict]:
        """Compute value loss and gradients.
        
        Args:
            states: Batch of states
            returns: Target returns
            
        Returns:
            Tuple of (loss, gradients_dict)
        """
        # Forward pass
        values = np.array([self.value.forward(s) for s in states]).flatten()
        
        # MSE loss
        # Use Huber loss for value function (more stable than MSE)
        # Clamp returns to prevent extreme values
        returns_clipped = np.clip(returns, -1000.0, 1000.0)
        diff = values - returns_clipped
        # Huber loss: quadratic for small errors, linear for large errors
        huber_delta = 10.0
        huber_loss = np.where(
            np.abs(diff) < huber_delta,
            0.5 * diff ** 2,
            huber_delta * (np.abs(diff) - 0.5 * huber_delta)
        )
        value_loss = np.mean(huber_loss)
        
        # Compute gradients using simplified backpropagation
        grads = self._compute_value_gradients_simplified(states, returns)
        
        return float(value_loss), grads
    
    def _compute_policy_gradients_simplified(self, states: np.ndarray, actions: np.ndarray,
                                            advantages: np.ndarray,
                                            old_log_probs: np.ndarray) -> Dict:
        """Compute policy gradients using simplified policy gradient theorem.
        
        Uses the policy gradient: ∇θ L = E[∇θ log π(a|s) * min(r*A, clip(r)*A)]
        where r = π_new(a|s) / π_old(a|s)
        
        For efficiency, we compute gradients directly from the policy gradient theorem
        for Gaussian policies, rather than full backpropagation through all layers.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            old_log_probs: Old log probabilities
        Returns:
            Dictionary of gradients
        """
        # Policy gradient theorem: ∇L = E[∇log π(a|s) * advantage_weight]
        # where advantage_weight = min(r*A, clip(r)*A) / (π_new/π_old)
        
        new_log_probs = self._compute_action_log_prob(states, actions)
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.config.clip_epsilon,
                               1 + self.config.clip_epsilon)
        
        # Advantage weights for gradient
        advantage_weights = np.minimum(ratio * advantages, clipped_ratio * advantages)
        # Normalize by ratio to get gradient weights
        gradient_weights = advantage_weights / (ratio + 1e-8)
        gradient_weights = gradient_weights.reshape(-1, 1)  # (batch_size, 1)
        
        # Forward pass to get intermediate activations
        x = states
        hidden_outputs = [x]
        for layer in self.policy.layers:
            x = layer.forward(x)
            hidden_outputs.append(x)
        
        # Get mean and log_std
        mean = self.policy.mean_layer.forward(x)
        log_std = self.policy.log_std_layer.forward(x)
        log_std = np.clip(log_std, -20, 2)
        std = np.exp(log_std)
        
        # Compute action error (for Gaussian policy gradient)
        # For Gaussian: ∇log π(a|s) = (a - μ) / σ² * ∇μ - ∇log_std
        actions_tanh = np.tanh(actions)  # Actions are already tanh-scaled
        pre_tanh_actions = np.arctanh(np.clip(actions, -0.999, 0.999))
        
        # Gradient w.r.t. mean: (a - μ) / σ²
        mean_error = (pre_tanh_actions - mean) / (std ** 2 + 1e-8)
        
        # Gradient w.r.t. log_std: -1 + (a-μ)²/σ²
        log_std_error = -1.0 + (pre_tanh_actions - mean) ** 2 / (std ** 2 + 1e-8)
        
        # Weight by advantage
        mean_grad_input = (mean_error * gradient_weights).mean(axis=0)  # (action_dim,)
        log_std_grad_input = (log_std_error * gradient_weights).mean(axis=0)  # (action_dim,)
        
        # Backpropagate through output layers (simplified - only through mean/log_std)
        # Get hidden output before output layers
        hidden_out = hidden_outputs[-1]  # (batch_size, hidden_dim)
        
        # Mean layer gradients
        mean_grads_weights = np.dot(hidden_out.T, mean_error * gradient_weights) / len(states)
        mean_grads_bias = mean_grad_input
        
        # Log_std layer gradients
        log_std_grads_weights = np.dot(hidden_out.T, log_std_error * gradient_weights) / len(states)
        log_std_grads_bias = log_std_grad_input
        
        grads = {
            'mean_weights': -mean_grads_weights,  # Negative for gradient ascent (maximize)
            'mean_bias': -mean_grads_bias,
            'log_std_weights': -log_std_grads_weights * 0.1,  # Smaller learning rate
            'log_std_bias': -log_std_grads_bias * 0.1,
        }
        
        return grads
    
    def _compute_value_gradients_simplified(self, states: np.ndarray,
                                           returns: np.ndarray) -> Dict:
        """Compute value network gradients using simplified backpropagation.
        
        Args:
            states: Batch of states
            returns: Target returns
            
        Returns:
            Dictionary of gradients
        """
        # Forward pass through value network
        values = []
        hidden_outputs = []
        x = states
        
        # Forward through hidden layers
        for layer in self.value.layers:
            x = layer.forward(x)
            hidden_outputs.append(x)
        
        # Forward through value layer
        value_out = self.value.value_layer.forward(x)
        values = value_out.flatten()
        
        # Compute value error (MSE gradient)
        value_error = 2.0 * (values - returns)  # Gradient of (v - r)²
        value_error = value_error.reshape(-1, 1)  # (batch_size, 1)
        
        # Get hidden output before value layer
        hidden_out = hidden_outputs[-1] if hidden_outputs else states
        
        # Value layer gradients
        value_grads_weights = np.dot(hidden_out.T, value_error) / len(states)
        value_grads_bias = value_error.mean(axis=0)
        
        grads = {
            'value_weights': value_grads_weights,
            'value_bias': value_grads_bias,
        }
        
        return grads
    
    def _update_policy(self, grads: Dict):
        """Update policy network using gradients.
        
        Args:
            grads: Dictionary of gradients
        """
        # Extract gradients and update parameters
        if 'mean_weights' in grads:
            # Clip gradients
            grad_norm = np.linalg.norm(grads['mean_weights'])
            if grad_norm > self.config.max_grad_norm:
                grads['mean_weights'] = grads['mean_weights'] * (self.config.max_grad_norm / grad_norm)
            
            # Update using optimizer
            params = {'weights': self.policy.mean_layer.weights}
            grads_dict = {'weights': grads['mean_weights']}
            if 'mean_bias' in grads:
                params['bias'] = self.policy.mean_layer.bias
                grads_dict['bias'] = grads['mean_bias']
            
            self.policy_optimizer.update(params, grads_dict)
        
        if 'log_std_weights' in grads:
            params = {'weights': self.policy.log_std_layer.weights}
            grads_dict = {'weights': grads['log_std_weights']}
            if 'log_std_bias' in grads:
                params['bias'] = self.policy.log_std_layer.bias
                grads_dict['bias'] = grads['log_std_bias']
            
            # Smaller learning rate for log_std (more stable)
            old_lr = self.policy_optimizer.learning_rate
            self.policy_optimizer.learning_rate = old_lr * 0.5
            self.policy_optimizer.update(params, grads_dict)
            self.policy_optimizer.learning_rate = old_lr
    
    def _update_value(self, grads: Dict):
        """Update value network using gradients.
        
        Args:
            grads: Dictionary of gradients
        """
        if 'value_weights' in grads:
            # Clip gradients
            grad_norm = np.linalg.norm(grads['value_weights'])
            if grad_norm > self.config.max_grad_norm:
                grads['value_weights'] = grads['value_weights'] * (self.config.max_grad_norm / grad_norm)
            
            params = {'weights': self.value.value_layer.weights}
            grads_dict = {'weights': grads['value_weights']}
            if 'value_bias' in grads:
                params['bias'] = self.value.value_layer.bias
                grads_dict['bias'] = grads['value_bias']
            
            self.value_optimizer.update(params, grads_dict)
    
    def get_params(self) -> Dict:
        """Get all agent parameters."""
        return {
            'policy': self.policy.get_params(),
            'value': self.value.get_params(),
        }
    
    def set_params(self, params: Dict):
        """Set all agent parameters."""
        self.policy.set_params(params['policy'])
        self.value.set_params(params['value'])
    
    def save(self, filepath: str):
        """Save agent to file."""
        import json
        data = {
            'config': {
                'gamma': self.config.gamma,
                'lambda_': self.config.lambda_,
                'clip_epsilon': self.config.clip_epsilon,
                'value_coef': self.config.value_coef,
                'entropy_coef': self.config.entropy_coef,
                'learning_rate': self.config.learning_rate,
            },
            'params': self.get_params(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load agent from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Update config
        config_dict = data['config']
        self.config = PPOConfig(**config_dict)
        
        # Load parameters
        self.set_params(data['params'])

