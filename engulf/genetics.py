"""Gene definitions, metadata, and effect systems for creature genetics.

This module provides a structured system for managing 64 different genes,
their metadata (ranges, mutation rates), and how they affect creature behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Any, Optional
import random


@dataclass
class GeneDef:
    """Definition and metadata for a single gene.
    
    Attributes:
        name: Gene name (internal identifier)
        category: Gene category (e.g., 'appearance', 'personality', 'ability')
        default_range: (min, max) tuple for default value range
        mutation_scale: How much this gene mutates during breeding
        description: Human-readable description
        default_value: Default value if not specified (None = random in range)
    """
    name: str
    category: str
    default_range: tuple[float, float]
    mutation_scale: float
    description: str
    default_value: Optional[float] = None


# Gene categories
CATEGORY_IDENTITY = "identity"
CATEGORY_APPEARANCE = "appearance"
CATEGORY_PERSONALITY = "personality"
CATEGORY_ABILITY = "ability"
CATEGORY_PERCEPTION = "perception"
CATEGORY_ENVIRONMENT = "environment"
CATEGORY_LIFE = "life"
CATEGORY_BREEDING = "breeding"
CATEGORY_RESOURCE = "resource"
CATEGORY_SOCIAL = "social"


# Define all 64 genes with their metadata
GENES: Dict[str, GeneDef] = {
    # === IDENTITY (1 gene) ===
    "species_id": GeneDef(
        name="species_id",
        category=CATEGORY_IDENTITY,
        default_range=(0, 2),  # 0..N-1, typically 0-2
        mutation_scale=0.0,  # Never mutates (inherited from parent)
        description="Species identifier"
    ),
    
    # === APPEARANCE (2 genes) ===
    "body_radius": GeneDef(
        name="body_radius",
        category=CATEGORY_APPEARANCE,
        default_range=(8.0, 14.0),  # Reduced max from 22.0 to 16.0
        mutation_scale=1.0,
        description="Physical body radius"
    ),
    "hue": GeneDef(
        name="hue",
        category=CATEGORY_APPEARANCE,
        default_range=(0.0, 360.0),
        mutation_scale=8.0,
        description="Color hue in degrees"
    ),
    
    # === PERSONALITY/BEHAVIOR (18 genes) ===
    "curiosity": GeneDef(
        name="curiosity",
        category=CATEGORY_PERSONALITY,
        default_range=(0.2, 0.9),
        mutation_scale=0.05,
        description="Exploration tendency"
    ),
    "aggression": GeneDef(
        name="aggression",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 0.8),
        mutation_scale=0.05,
        description="Aggressive behavior"
    ),
    "caution": GeneDef(
        name="caution",
        category=CATEGORY_PERSONALITY,
        default_range=(0.1, 0.9),
        mutation_scale=0.05,
        description="Caution level"
    ),
    "sociability": GeneDef(
        name="sociability",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to approach same species"
    ),
    "territoriality": GeneDef(
        name="territoriality",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Desire to monopolize resource areas"
    ),
    "strategic_flexibility": GeneDef(
        name="strategic_flexibility",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to adjust strategy based on environment"
    ),
    "predatory_instinct": GeneDef(
        name="predatory_instinct",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to actively hunt smaller creatures"
    ),
    "exploration_drive": GeneDef(
        name="exploration_drive",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Desire to explore unknown areas"
    ),
    "risk_tolerance": GeneDef(
        name="risk_tolerance",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Willingness to take risks for rewards"
    ),
    "hoarding_instinct": GeneDef(
        name="hoarding_instinct",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to stay longer in resource-rich areas"
    ),
    "migration_tendency": GeneDef(
        name="migration_tendency",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to regularly move and find new areas"
    ),
    "competitiveness": GeneDef(
        name="competitiveness",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Performance in resource competition"
    ),
    "breeding_urge": GeneDef(
        name="breeding_urge",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Strong desire to find breeding opportunities"
    ),
    "group_defense": GeneDef(
        name="group_defense",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to defend group members from threats"
    ),
    "resource_competition_strategy": GeneDef(
        name="resource_competition_strategy",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Resource competition strategy (0=avoid, 1=compete)"
    ),
    "group_size_preference": GeneDef(
        name="group_size_preference",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Preferred group size"
    ),
    "fear_threshold": GeneDef(
        name="fear_threshold",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Danger level that triggers avoidance"
    ),
    "stress_resistance": GeneDef(
        name="stress_resistance",
        category=CATEGORY_PERSONALITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to maintain performance under stress"
    ),
    
    # === PERCEPTION/COGNITION (8 genes) ===
    "perception_range": GeneDef(
        name="perception_range",
        category=CATEGORY_PERCEPTION,
        default_range=(0.8, 1.5),
        mutation_scale=0.08,
        description="Multiplier for resource/hazard detection range"
    ),
    "memory": GeneDef(
        name="memory",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to remember terrain experiences"
    ),
    "olfactory_sensitivity": GeneDef(
        name="olfactory_sensitivity",
        category=CATEGORY_PERCEPTION,
        default_range=(0.7, 1.4),
        mutation_scale=0.08,
        description="Sensitivity to resource scents"
    ),
    "learning_rate": GeneDef(
        name="learning_rate",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Speed of learning from experiences"
    ),
    "time_awareness": GeneDef(
        name="time_awareness",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Awareness of time passage and seasons"
    ),
    "terrain_memory_strength": GeneDef(
        name="terrain_memory_strength",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Persistence of terrain memories"
    ),
    "spatial_orientation": GeneDef(
        name="spatial_orientation",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to navigate and remember locations"
    ),
    "environmental_prediction": GeneDef(
        name="environmental_prediction",
        category=CATEGORY_PERCEPTION,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to predict environmental changes"
    ),
    
    # === PHYSICAL ABILITY (6 genes) ===
    "max_speed": GeneDef(
        name="max_speed",
        category=CATEGORY_ABILITY,
        default_range=(110.0, 220.0),
        mutation_scale=6.0,
        description="Maximum speed in px/s"
    ),
    "angular_speed": GeneDef(
        name="angular_speed",
        category=CATEGORY_ABILITY,
        default_range=(1.2, 4.5),
        mutation_scale=0.2,
        description="Angular velocity in rad/s"
    ),
    "agility": GeneDef(
        name="agility",
        category=CATEGORY_ABILITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Quick direction change ability"
    ),
    "stamina": GeneDef(
        name="stamina",
        category=CATEGORY_ABILITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Sustained movement capability"
    ),
    "group_cohesion": GeneDef(
        name="group_cohesion",
        category=CATEGORY_ABILITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Cohesion within groups"
    ),
    "camouflage": GeneDef(
        name="camouflage",
        category=CATEGORY_ABILITY,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Reduced visibility to predators"
    ),
    
    # === ENVIRONMENTAL ADAPTATION (9 genes) ===
    "environmental_adaptability": GeneDef(
        name="environmental_adaptability",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Adaptation speed to terrain changes"
    ),
    "cold_resistance": GeneDef(
        name="cold_resistance",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Resistance to cold terrain effects"
    ),
    "heat_resistance": GeneDef(
        name="heat_resistance",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Resistance to heat terrain effects"
    ),
    "aquatic_affinity": GeneDef(
        name="aquatic_affinity",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Adaptability to water terrain"
    ),
    "nocturnality": GeneDef(
        name="nocturnality",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Activity in low-light environments"
    ),
    "niche_breadth": GeneDef(
        name="niche_breadth",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to adapt to different environments"
    ),
    "terrain_specialization": GeneDef(
        name="terrain_specialization",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Specialization in specific terrain types"
    ),
    "terrain_transition_adaptation": GeneDef(
        name="terrain_transition_adaptation",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Quick adaptation when switching terrains"
    ),
    "toxin_resistance": GeneDef(
        name="toxin_resistance",
        category=CATEGORY_ENVIRONMENT,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Resistance to toxic terrain"
    ),
    
    # === LIFE (3 genes) ===
    "lifespan_secs": GeneDef(
        name="lifespan_secs",
        category=CATEGORY_LIFE,
        default_range=(35.0, 80.0),  # Increased from (25.0, 60.0) to (35.0, 80.0) for longer lifespans
        mutation_scale=2.5,
        description="Lifespan in seconds"
    ),
    "recovery_rate": GeneDef(
        name="recovery_rate",
        category=CATEGORY_LIFE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Recovery speed from negative effects"
    ),
    "immune_system": GeneDef(
        name="immune_system",
        category=CATEGORY_LIFE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Resistance to continuous environmental damage"
    ),
    
    # === BREEDING (6 genes) ===
    "fertility": GeneDef(
        name="fertility",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Breeding cooldown modifier"
    ),
    "beauty": GeneDef(
        name="beauty",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Breeding success modifier"
    ),
    "mutation_rate": GeneDef(
        name="mutation_rate",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Gene mutation rate in offspring"
    ),
    "offspring_quality": GeneDef(
        name="offspring_quality",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Initial attributes of offspring"
    ),
    "mate_selectivity": GeneDef(
        name="mate_selectivity",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Mate selection standards"
    ),
    "breeding_timing": GeneDef(
        name="breeding_timing",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to choose optimal breeding timing"
    ),
    "evolutionary_plasticity": GeneDef(
        name="evolutionary_plasticity",
        category=CATEGORY_BREEDING,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Evolutionary potential (offspring variation range)"
    ),
    
    # === RESOURCE UTILIZATION (9 genes) ===
    "metabolism": GeneDef(
        name="metabolism",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Growth rate modifier"
    ),
    "speed_adaptability": GeneDef(
        name="speed_adaptability",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Speed boost modifier from resources"
    ),
    "energy_efficiency": GeneDef(
        name="energy_efficiency",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Energy consumption efficiency"
    ),
    "body_plasticity": GeneDef(
        name="body_plasticity",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Body adaptability to environmental changes"
    ),
    "resilience": GeneDef(
        name="resilience",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Hazard death resistance"
    ),
    "resource_conversion_efficiency": GeneDef(
        name="resource_conversion_efficiency",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Efficiency of converting resources to attributes"
    ),
    "energy_storage": GeneDef(
        name="energy_storage",
        category=CATEGORY_RESOURCE,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Ability to store extra energy"
    ),
    
    # === SOCIAL (3 genes) ===
    "leadership": GeneDef(
        name="leadership",
        category=CATEGORY_SOCIAL,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Influence within groups"
    ),
    "information_sharing": GeneDef(
        name="information_sharing",
        category=CATEGORY_SOCIAL,
        default_range=(0.0, 1.0),
        mutation_scale=0.08,
        description="Tendency to share terrain information"
    ),
    "exploration_radius": GeneDef(
        name="exploration_radius",
        category=CATEGORY_SOCIAL,
        default_range=(0.6, 1.4),
        mutation_scale=0.08,
        description="Willingness to explore distant areas"
    ),
}


def get_gene_def(gene_name: str) -> Optional[GeneDef]:
    """Get gene definition by name.
    
    Args:
        gene_name: Name of the gene
        
    Returns:
        GeneDef if found, None otherwise
    """
    return GENES.get(gene_name)


def get_genes_by_category(category: str) -> Dict[str, GeneDef]:
    """Get all genes in a category.
    
    Args:
        category: Category name
        
    Returns:
        Dictionary of gene name -> GeneDef
    """
    return {name: gene for name, gene in GENES.items() if gene.category == category}


def are_different_species(genome_a: Dict[str, float], genome_b: Dict[str, float], 
                         threshold: int = 5) -> bool:
    """Check if two genomes belong to different species based on gene differences.
    
    Two creatures are considered different species if they have at least `threshold`
    genes that are significantly different (more than 0.3 difference).
    
    Args:
        genome_a: First genome dictionary
        genome_b: Second genome dictionary
        threshold: Minimum number of different genes to be considered different species (default: 5)
        
    Returns:
        True if creatures are different species, False if same species
    """
    different_count = 0
    
    # Check key distinguishing genes (excluding species_id which is no longer used)
    key_genes = [
        'curiosity', 'aggression', 'caution', 'sociability', 'territoriality',
        'strategic_flexibility', 'predatory_instinct', 'exploration_drive',
        'risk_tolerance', 'hoarding_instinct', 'migration_tendency',
        'body_radius', 'hue', 'max_speed', 'angular_speed',
        'agility', 'stamina', 'resilience', 'metabolism', 'fertility'
    ]
    
    for gene_name in key_genes:
        val_a = genome_a.get(gene_name, 0.0)
        val_b = genome_b.get(gene_name, 0.0)
        
        # Check if genes differ significantly (threshold: 0.3)
        if abs(val_a - val_b) > 0.3:
            different_count += 1
            if different_count >= threshold:
                return True
    
    # If not enough differences in key genes, check all genes
    for gene_name in GENES.keys():
        if gene_name == 'species_id':
            continue
        
        val_a = genome_a.get(gene_name, 0.0)
        val_b = genome_b.get(gene_name, 0.0)
        
        if abs(val_a - val_b) > 0.3:
            different_count += 1
            if different_count >= threshold:
                return True
    
    return different_count >= threshold


def is_same_species(genome_a: Dict[str, float], genome_b: Dict[str, float]) -> bool:
    """Check if two genomes belong to the same species.
    
    Args:
        genome_a: First genome dictionary
        genome_b: Second genome dictionary
        
    Returns:
        True if same species, False otherwise
    """
    return not are_different_species(genome_a, genome_b, threshold=5)


def calculate_genetic_distance(genome_a: Dict[str, float], genome_b: Dict[str, float]) -> float:
    """Calculate genetic distance between two genomes.
    
    Uses weighted Euclidean distance on normalized gene values.
    Key genes (personality, behavior) are weighted more heavily.
    
    Args:
        genome_a: First genome dictionary
        genome_b: Second genome dictionary
        
    Returns:
        Genetic distance (0.0 = identical, higher = more different)
    """
    # Key distinguishing genes with higher weights
    key_genes = {
        # Personality/behavior genes (weight 2.0)
        'curiosity': 2.0, 'aggression': 2.0, 'caution': 2.0,
        'sociability': 2.0, 'exploration_drive': 2.0, 'predatory_instinct': 2.0,
        'risk_tolerance': 2.0, 'competitiveness': 2.0,
        # Physical genes (weight 1.5)
        'max_speed': 1.5, 'angular_speed': 1.5, 'body_radius': 1.5,
        'agility': 1.5, 'stamina': 1.5,
        # Other genes (weight 1.0)
    }
    
    total_distance = 0.0
    total_weight = 0.0
    
    # Calculate weighted distance for key genes
    for gene_name, weight in key_genes.items():
        if gene_name in genome_a and gene_name in genome_b:
            val_a = genome_a[gene_name]
            val_b = genome_b[gene_name]
            # Normalize by gene range
            gene_def = GENES.get(gene_name)
            if gene_def:
                min_val, max_val = gene_def.default_range
                range_size = max(0.001, max_val - min_val)  # Avoid division by zero
                normalized_diff = abs(val_a - val_b) / range_size
                total_distance += normalized_diff * weight
                total_weight += weight
    
    # Calculate distance for remaining genes (weight 1.0)
    for gene_name, gene_def in GENES.items():
        if gene_name in ('species_id', 'hue') or gene_name in key_genes:
            continue
        if gene_name in genome_a and gene_name in genome_b:
            val_a = genome_a[gene_name]
            val_b = genome_b[gene_name]
            min_val, max_val = gene_def.default_range
            range_size = max(0.001, max_val - min_val)
            normalized_diff = abs(val_a - val_b) / range_size
            total_distance += normalized_diff * 1.0
            total_weight += 1.0
    
    # Average weighted distance
    if total_weight > 0:
        avg_distance = total_distance / total_weight
    else:
        avg_distance = 0.0
    
    return avg_distance


def check_species_divergence(offspring: Dict[str, float], parent_a: Dict[str, float],
                            parent_b: Dict[str, float], divergence_threshold: float = 0.35) -> bool:
    """Check if offspring is divergent enough to be a new species.
    
    Args:
        offspring: Offspring genome dictionary
        parent_a: First parent genome dictionary
        parent_b: Second parent genome dictionary
        divergence_threshold: Minimum genetic distance to be considered new species (default: 0.35)
        
    Returns:
        True if offspring should be considered a new species
    """
    # Calculate distance from both parents
    dist_from_a = calculate_genetic_distance(offspring, parent_a)
    dist_from_b = calculate_genetic_distance(offspring, parent_b)
    
    # Average distance from parents
    avg_distance = (dist_from_a + dist_from_b) / 2.0
    
    # Also check distance between parents (to ensure mutation, not just different parents)
    parent_distance = calculate_genetic_distance(parent_a, parent_b)
    
    # Offspring is new species if:
    # 1. Average distance from parents > threshold, AND
    # 2. Offspring is significantly different from both parents
    return (avg_distance > divergence_threshold and
            dist_from_a > divergence_threshold * 0.8 and
            dist_from_b > divergence_threshold * 0.8)


def generate_random_genome(num_species: int = 3) -> Dict[str, float]:
    """Generate a random genome with all genes.
    
    Args:
        num_species: Number of species to choose from (deprecated, kept for compatibility)
        
    Returns:
        Dictionary mapping gene name -> value
    """
    genome = {}
    
    # Special handling for species_id (deprecated but kept for compatibility)
    genome["species_id"] = float(random.randrange(num_species))
    
    # Generate hue randomly
    genome["hue"] = random.uniform(0.0, 360.0)
    
    # Generate all other genes
    for name, gene_def in GENES.items():
        if name in ("species_id", "hue"):
            continue
        min_val, max_val = gene_def.default_range
        genome[name] = random.uniform(min_val, max_val)
    
    return genome


def generate_base_species_genome(species_index: int, high_speed: bool = True) -> Dict[str, float]:
    """Generate a base genome for one of three initial species.
    
    Args:
        species_index: Which species (0, 1, or 2)
        high_speed: Whether to give high initial speed (default: True)
        
    Returns:
        Dictionary mapping gene name -> value
    """
    genome = {}
    
    # Set species_id (deprecated but kept for compatibility)
    genome["species_id"] = float(species_index)
    
    # Each species has distinct characteristics with high activity
    if species_index == 0:
        # Species 0: Aggressive, very fast, high metabolism
        base_genes = {
            "hue": 0.0,  # Red
            "aggression": 0.8,
            "curiosity": 0.7,  # Increased from 0.6
            "caution": 0.3,
            "predatory_instinct": 0.9,
            "sociability": 0.4,
            "max_speed": 280.0 if high_speed else 200.0,  # Increased from 250
            "angular_speed": 5.0,  # Increased from 4.0
            "agility": 0.8,  # Added for higher activity
            "stamina": 0.8,  # Added for sustained movement
            "metabolism": 1.2,
            "body_radius": 16.0,
            "lifespan_secs": 50.0,  # Increased from default for better survival
        }
    elif species_index == 1:
        # Species 1: Cautious, very fast, high exploration
        base_genes = {
            "hue": 120.0,  # Green
            "aggression": 0.2,
            "curiosity": 0.95,  # Increased from 0.9
            "caution": 0.8,
            "exploration_drive": 0.95,  # Increased from 0.9
            "sociability": 0.7,
            "max_speed": 270.0 if high_speed else 195.0,  # Increased from 240
            "angular_speed": 4.5,  # Increased from 3.5
            "agility": 0.9,  # Added for higher activity
            "stamina": 0.85,  # Added for sustained movement
            "metabolism": 1.0,
            "body_radius": 14.0,
            "lifespan_secs": 55.0,  # Increased from default for better survival
        }
    else:  # species_index == 2
        # Species 2: Social, very fast, high fertility
        base_genes = {
            "hue": 240.0,  # Blue
            "aggression": 0.3,
            "curiosity": 0.6,  # Increased from 0.5
            "caution": 0.6,
            "sociability": 0.9,
            "fertility": 1.2,
            "max_speed": 260.0 if high_speed else 185.0,  # Increased from 230
            "angular_speed": 4.8,  # Increased from 3.8
            "agility": 0.75,  # Added for higher activity
            "stamina": 0.75,  # Added for sustained movement
            "metabolism": 0.9,
            "body_radius": 15.0,
            "lifespan_secs": 60.0,  # Increased from default for better survival (social species live longer)
        }
    
    # Start with base genes
    genome.update(base_genes)
    
    # Generate remaining genes with some variation
    for name, gene_def in GENES.items():
        if name in base_genes or name == "species_id":
            continue
        
        min_val, max_val = gene_def.default_range
        
        # Add some variation to make each species member slightly different
        if species_index == 0 and name in ["territoriality", "competitiveness"]:
            # Species 0 tends toward high territoriality
            value = random.uniform(max(min_val, 0.6), max_val)
        elif species_index == 1 and name in ["exploration_drive", "risk_tolerance"]:
            # Species 1 tends toward high exploration
            value = random.uniform(max(min_val, 0.5), max_val)
        elif species_index == 2 and name in ["fertility", "group_cohesion"]:
            # Species 2 tends toward high social traits
            value = random.uniform(max(min_val, 0.6), max_val)
        else:
            value = random.uniform(min_val, max_val)
        
        genome[name] = value
    
    return genome


def breed_genomes(parent_a: Dict[str, float], parent_b: Dict[str, float],
                  fitness_a: float = 0.5, fitness_b: float = 0.5) -> Dict[str, float]:
    """Breed two genomes with mutations and fitness-weighted inheritance.
    
    Args:
        parent_a: First parent genome
        parent_b: Second parent genome (must have same species_id)
        fitness_a: Fitness score of parent A (0.0-1.0, normalized)
        fitness_b: Fitness score of parent B (0.0-1.0, normalized)
        
    Returns:
        New genome with mixed traits and mutations
    """
    def mix_value(
        val_a: float,
        val_b: float,
        gene_def: GeneDef,
        mutation_boost: float = 1.0,
        fitness_weight_a: float = 0.5,
        fitness_weight_b: float = 0.5
    ) -> float:
        """Mix two gene values with mutation and fitness weighting.
        
        Args:
            val_a: Value from parent A
            val_b: Value from parent B
            gene_def: Gene definition
            mutation_boost: Multiplier for mutation (from evolutionary_plasticity)
            fitness_weight_a: Weight for parent A (based on fitness)
            fitness_weight_b: Weight for parent B (based on fitness)
            
        Returns:
            Mixed and mutated value
        """
        # Fitness-weighted average (higher fitness = more influence)
        total_fitness = fitness_weight_a + fitness_weight_b
        if total_fitness > 0:
            weight_a = fitness_weight_a / total_fitness
            weight_b = fitness_weight_b / total_fitness
        else:
            weight_a = weight_b = 0.5
        
        v = weight_a * val_a + weight_b * val_b
        # Add mutation
        mut_scale = gene_def.mutation_scale * mutation_boost
        v += random.uniform(-1.0, 1.0) * mut_scale
        
        # Clamp to range
        min_val, max_val = gene_def.default_range
        return max(min_val, min(max_val, v))
    
    # Normalize fitness scores (avoid division by zero)
    total_fitness = fitness_a + fitness_b
    if total_fitness > 0:
        fitness_weight_a = fitness_a / total_fitness
        fitness_weight_b = fitness_b / total_fitness
    else:
        fitness_weight_a = fitness_weight_b = 0.5
    
    # Get mutation boost from parents' evolutionary_plasticity
    mut_boost_a = parent_a.get("evolutionary_plasticity", 1.0)
    mut_boost_b = parent_b.get("evolutionary_plasticity", 1.0)
    mutation_boost = 1.0 + 0.5 * (mut_boost_a + mut_boost_b)  # 1.0 to 2.0
    
    offspring = {}
    
    # Special handling for species_id (never mutates, inherited from parent)
    offspring["species_id"] = parent_a["species_id"]
    
    # Special handling for hue (circular, fitness-weighted)
    hue_a = parent_a["hue"]
    hue_b = parent_b["hue"]
    hue_avg = (fitness_weight_a * hue_a + fitness_weight_b * hue_b) % 360.0
    hue = (hue_avg + random.uniform(-8.0, 8.0) * mutation_boost) % 360.0
    offspring["hue"] = hue
    
    # Mix all other genes with fitness weighting
    for name, gene_def in GENES.items():
        if name in ("species_id", "hue"):
            continue
        val_a = parent_a.get(name, gene_def.default_range[0])
        val_b = parent_b.get(name, gene_def.default_range[0])
        offspring[name] = mix_value(
            val_a, val_b, gene_def, mutation_boost,
            fitness_weight_a, fitness_weight_b
        )
    
    return offspring


# Gene effect system: define how genes affect creature behavior
class GeneEffects:
    """System for applying gene effects to creature behavior.
    
    This class provides methods to calculate various behavioral modifiers
    and traits based on gene values.
    """
    
    @staticmethod
    def get_perception_multiplier(genome: Dict[str, float]) -> float:
        """Get perception range multiplier.
        
        Args:
            genome: Creature genome
            
        Returns:
            Multiplier for perception range (0.8-1.5)
        """
        return genome.get("perception_range", 1.0)
    
    @staticmethod
    def get_effective_speed(genome: Dict[str, float], base_speed_mult: float) -> float:
        """Get effective speed considering genes.
        
        Args:
            genome: Creature genome
            base_speed_mult: Base speed multiplier from terrain
            
        Returns:
            Effective speed multiplier
        """
        max_speed = genome.get("max_speed", 165.0)
        stamina = genome.get("stamina", 0.5)
        agility = genome.get("agility", 0.5)
        
        # Stamina affects sustained speed
        stamina_effect = 0.8 + 0.2 * stamina
        
        # Agility affects speed in complex terrain
        agility_effect = 0.9 + 0.1 * agility
        
        return base_speed_mult * stamina_effect * agility_effect
    
    @staticmethod
    def get_resource_attraction_strength(genome: Dict[str, float]) -> float:
        """Get resource attraction strength.
        
        Args:
            genome: Creature genome
            
        Returns:
            Attraction strength multiplier
        """
        curiosity = genome.get("curiosity", 0.5)
        olfactory = genome.get("olfactory_sensitivity", 1.0)
        hoarding = genome.get("hoarding_instinct", 0.5)
        
        return (1.0 + curiosity * 0.5) * olfactory * (1.0 + hoarding * 0.3)
    
    @staticmethod
    def get_hazard_repulsion_strength(genome: Dict[str, float]) -> float:
        """Get hazard repulsion strength.
        
        Args:
            genome: Creature genome
            
        Returns:
            Repulsion strength multiplier
        """
        caution = genome.get("caution", 0.5)
        fear_threshold = genome.get("fear_threshold", 0.5)
        risk_tolerance = genome.get("risk_tolerance", 0.5)
        
        # Higher caution and lower fear threshold = stronger repulsion
        # Lower risk tolerance = stronger repulsion
        base_repulsion = 1.0 + caution * 0.5
        threshold_effect = 1.0 + (1.0 - fear_threshold) * 0.3
        risk_effect = 1.0 + (1.0 - risk_tolerance) * 0.2
        
        return base_repulsion * threshold_effect * risk_effect
    
    @staticmethod
    def get_environmental_resistance(genome: Dict[str, float], terrain_type: str) -> float:
        """Get resistance to specific terrain hazards.
        
        Args:
            genome: Creature genome
            terrain_type: Type of terrain
            
        Returns:
            Resistance multiplier (higher = less affected)
        """
        resistance_map = {
            "ice": "cold_resistance",
            "lava": "heat_resistance",
            "geyser": "heat_resistance",
            "water": "aquatic_affinity",
            "toxic": "toxin_resistance",
            "storm": "immune_system",
        }
        
        resistance_gene = resistance_map.get(terrain_type, "resilience")
        resistance = genome.get(resistance_gene, 0.5)
        
        # Base resilience always applies
        base_resilience = genome.get("resilience", 0.5)
        
        return 0.5 + 0.5 * (resistance + base_resilience) / 2.0
    
    @staticmethod
    def get_breeding_success_modifier(genome: Dict[str, float]) -> float:
        """Get breeding success probability modifier.
        
        Args:
            genome: Creature genome
            
        Returns:
            Success probability multiplier
        """
        beauty = genome.get("beauty", 0.5)
        fertility = genome.get("fertility", 0.5)
        breeding_timing = genome.get("breeding_timing", 0.5)
        
        return 0.7 + 0.3 * (beauty * 0.4 + fertility * 0.4 + breeding_timing * 0.2)
    
    @staticmethod
    def get_learning_speed(genome: Dict[str, float]) -> float:
        """Get learning speed for terrain memory.
        
        Args:
            genome: Creature genome
            
        Returns:
            Learning speed multiplier
        """
        learning_rate = genome.get("learning_rate", 0.5)
        memory = genome.get("memory", 0.5)
        
        return 0.5 + 0.5 * (learning_rate + memory) / 2.0
    
    @staticmethod
    def get_group_attraction(genome: Dict[str, float]) -> float:
        """Get attraction to groups of same species.
        
        Args:
            genome: Creature genome
            
        Returns:
            Group attraction strength
        """
        sociability = genome.get("sociability", 0.5)
        group_cohesion = genome.get("group_cohesion", 0.5)
        
        return 0.3 + 0.7 * (sociability + group_cohesion) / 2.0

