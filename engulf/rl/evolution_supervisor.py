"""Evolution Supervisor for RL-based evolutionary learning.

This module implements evolutionary mechanisms on top of RL:
- Fitness-based selection
- Genetic inheritance with mutations
- Population diversity tracking
- Evolutionary pressure
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class FitnessMetrics:
    """Fitness metrics for a creature."""
    total_reward: float = 0.0
    lifespan: float = 0.0
    offspring_count: int = 0
    exploration_score: float = 0.0
    resource_score: float = 0.0
    survival_bonus: float = 0.0
    
    @property
    def fitness(self) -> float:
        """Composite fitness score."""
        # Weighted combination of metrics
        return (
            self.total_reward * 1.0 +
            self.lifespan * 0.5 +
            self.offspring_count * 50.0 +
            self.exploration_score * 10.0 +
            self.resource_score * 20.0 +
            self.survival_bonus * 5.0
        )


class EvolutionSupervisor:
    """Supervises evolutionary processes for RL creatures.
    
    Provides:
    - Fitness tracking
    - Natural selection pressure
    - Genetic diversity monitoring
    - Population health metrics
    """
    
    def __init__(self, selection_rate: float = 0.2, 
                 mutation_rate: float = 0.1,
                 diversity_preservation: float = 0.1):
        """Initialize evolution supervisor.
        
        Args:
            selection_rate: Fraction of population selected for reproduction (0.2 = top 20%)
            mutation_rate: Probability of mutation per gene (0.1 = 10%)
            diversity_preservation: Fraction of random survivors to preserve diversity (0.1 = 10%)
        """
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.diversity_preservation = diversity_preservation
        
        # Fitness tracking
        self.creature_fitness: Dict[int, FitnessMetrics] = {}
        
        # Evolution statistics
        self.generation = 0
        self.stats_history: List[Dict] = []
        
        # Diversity tracking
        self._genetic_diversity_cache = None
        
    def track_fitness(self, creature_id: int, metrics: FitnessMetrics):
        """Track fitness metrics for a creature.
        
        Args:
            creature_id: Unique creature identifier
            metrics: Fitness metrics
        """
        self.creature_fitness[creature_id] = metrics
    
    def get_fitness(self, creature_id: int) -> float:
        """Get fitness score for a creature.
        
        Args:
            creature_id: Unique creature identifier
            
        Returns:
            Fitness score (0.0 if not tracked)
        """
        if creature_id in self.creature_fitness:
            return self.creature_fitness[creature_id].fitness
        return 0.0
    
    def select_parents(self, creatures: List, num_parents: int = None) -> List:
        """Select parents based on fitness.
        
        Uses fitness-based selection with diversity preservation.
        
        Args:
            creatures: List of creature objects
            num_parents: Number of parents to select (None = selection_rate * population)
            
        Returns:
            List of selected parent creatures
        """
        if not creatures:
            return []
        
        # Calculate fitness for all creatures
        fitness_scores = []
        for creature in creatures:
            if hasattr(creature, 'id'):
                fitness = self.get_fitness(creature.id)
            else:
                fitness = self.get_fitness(id(creature))
            fitness_scores.append((creature, fitness))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers (selection_rate)
        if num_parents is None:
            num_parents = max(1, int(len(creatures) * self.selection_rate))
        
        selected = [creature for creature, _ in fitness_scores[:num_parents]]
        
        # Add diversity preservation: random survivors
        num_diverse = max(1, int(len(creatures) * self.diversity_preservation))
        remaining = [creature for creature, _ in fitness_scores[num_parents:]]
        if remaining:
            diverse = random.sample(remaining, min(num_diverse, len(remaining)))
            selected.extend(diverse)
        
        return selected
    
    def calculate_genetic_diversity(self, creatures: List) -> float:
        """Calculate genetic diversity of population.
        
        Uses variance in key genes as diversity measure.
        
        Args:
            creatures: List of creatures
            
        Returns:
            Diversity score (0.0-1.0, higher = more diverse)
        """
        if len(creatures) < 2:
            return 1.0  # Single creature = maximum diversity
        
        # Extract key genes for diversity calculation
        key_genes = [
            'curiosity', 'exploration_drive', 'caution',
            'max_speed', 'aggression', 'learning_rate'
        ]
        
        gene_values = {gene: [] for gene in key_genes}
        
        for creature in creatures:
            if hasattr(creature, 'genome'):
                genome_dict = creature.genome.to_dict()
                for gene in key_genes:
                    if gene in genome_dict:
                        gene_values[gene].append(genome_dict[gene])
        
        # Calculate variance for each gene (normalized)
        variances = []
        for gene, values in gene_values.items():
            if len(values) > 1:
                variance = np.var(values)
                # Normalize by range (assuming 0-1 for most genes)
                max_var = 0.25  # Maximum variance for 0-1 range
                normalized_var = min(1.0, variance / max_var) if max_var > 0 else 0.0
                variances.append(normalized_var)
        
        # Average diversity across genes
        diversity = np.mean(variances) if variances else 0.5
        return float(diversity)
    
    def should_apply_selection_pressure(self, population_size: int, 
                                       target_size: int) -> bool:
        """Determine if natural selection should be applied.
        
        Args:
            population_size: Current population size
            target_size: Target population size
            
        Returns:
            True if selection should be applied
        """
        # Apply selection if population exceeds target by 20%
        return population_size > target_size * 1.2
    
    def get_population_health(self, creatures: List) -> Dict:
        """Get population health metrics.
        
        Args:
            creatures: List of creatures
            
        Returns:
            Dictionary of health metrics
        """
        if not creatures:
            return {
                'size': 0,
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'diversity': 0.0,
                'avg_lifespan': 0.0,
                'avg_offspring': 0.0,
            }
        
        # Collect metrics
        fitnesses = []
        lifespans = []
        offspring_counts = []
        
        for creature in creatures:
            if hasattr(creature, 'id'):
                creature_id = creature.id
            else:
                creature_id = id(creature)
            
            metrics = self.creature_fitness.get(creature_id, FitnessMetrics())
            fitnesses.append(metrics.fitness)
            lifespans.append(metrics.lifespan)
            offspring_counts.append(metrics.offspring_count)
        
        diversity = self.calculate_genetic_diversity(creatures)
        
        return {
            'size': len(creatures),
            'avg_fitness': float(np.mean(fitnesses)) if fitnesses else 0.0,
            'max_fitness': float(np.max(fitnesses)) if fitnesses else 0.0,
            'min_fitness': float(np.min(fitnesses)) if fitnesses else 0.0,
            'diversity': diversity,
            'avg_lifespan': float(np.mean(lifespans)) if lifespans else 0.0,
            'avg_offspring': float(np.mean(offspring_counts)) if offspring_counts else 0.0,
        }
    
    def record_generation(self, creatures: List):
        """Record generation statistics.
        
        Args:
            creatures: List of creatures in current generation
        """
        health = self.get_population_health(creatures)
        health['generation'] = self.generation
        self.stats_history.append(health)
        
        # Keep only last 100 generations
        if len(self.stats_history) > 100:
            self.stats_history = self.stats_history[-100:]
        
        self.generation += 1
    
    def get_evolution_stats(self) -> Dict:
        """Get evolution statistics.
        
        Returns:
            Dictionary of evolution statistics
        """
        if not self.stats_history:
            return {
                'generation': 0,
                'avg_fitness_trend': [],
                'diversity_trend': [],
            }
        
        recent = self.stats_history[-10:]  # Last 10 generations
        
        return {
            'generation': self.generation,
            'current_fitness': recent[-1]['avg_fitness'] if recent else 0.0,
            'fitness_trend': [s['avg_fitness'] for s in recent],
            'diversity_trend': [s['diversity'] for s in recent],
            'lifespan_trend': [s['avg_lifespan'] for s in recent],
        }

