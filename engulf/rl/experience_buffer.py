"""Experience buffer for storing and sampling training data."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random


class Experience:
    """Single experience tuple."""
    
    def __init__(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool, info: Optional[Dict] = None):
        """Initialize experience.
        
        Args:
            state: State array
            action: Action array
            reward: Reward value
            next_state: Next state array
            done: Done flag
            info: Optional additional info
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}


class ExperienceBuffer:
    """Simple FIFO experience buffer."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.size = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer.
        
        Args:
            experience: Experience to add
        """
        self.buffer.append(experience)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def sample_sequential(self, batch_size: int) -> List[Experience]:
        """Sample sequential batch (for on-policy methods).
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sequential experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        start_idx = random.randint(0, len(self.buffer) - batch_size)
        return [self.buffer[i] for i in range(start_idx, start_idx + batch_size)]
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.size = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedBuffer:
    """Prioritized experience replay buffer.
    
    Samples experiences with probability proportional to their priority (TD error).
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """Initialize prioritized buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / 100000  # Gradually increase beta to 1
        
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.size = 0
    
    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience with priority.
        
        Args:
            experience: Experience to add
            priority: Priority value (None = use max priority)
        """
        if priority is None:
            priority = self.max_priority
        
        idx = len(self.buffer)
        self.buffer.append(experience)
        self.priorities[idx] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Compute sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                  replace=False, p=probs)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices.
        
        Args:
            indices: Indices to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.buffer):
                self.priorities[idx] = (priority ** self.alpha)
                self.max_priority = max(self.max_priority, priority)
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.priorities.fill(0)
        self.max_priority = 1.0
        self.size = 0
    
    def __len__(self) -> int:
        return len(self.buffer)

