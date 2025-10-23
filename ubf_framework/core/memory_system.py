"""
Universal Behavioral Framework - Memory System

Implements persistent learning through experience memory storage and retrieval.
Agents form memories of significant events and use them to influence future decisions.
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class InteractionType(Enum):
    """Types of interactions that can form memories."""
    EXPLORATION = "exploration"
    SOCIAL = "social"
    COMBAT = "combat"
    ECONOMIC = "economic"
    LEARNING = "learning"
    SURVIVAL = "survival"
    CREATIVE = "creative"


@dataclass
class MemoryContext:
    """Context information that determines memory significance."""
    goal_relevance: float = 0.0      # 0.0-1.0: Related to active goals?
    novelty_factor: float = 0.0      # 0.0-1.0: First-time experience?
    social_importance: float = 0.0   # 0.0-1.0: Important relationships?
    survival_relevance: float = 0.0  # 0.0-1.0: Life-or-death situation?
    participants: List[str] = field(default_factory=list)  # Who else was involved?
    location: Optional[str] = None   # Where it happened
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_relevance': self.goal_relevance,
            'novelty_factor': self.novelty_factor,
            'social_importance': self.social_importance,
            'survival_relevance': self.survival_relevance,
            'participants': self.participants,
            'location': self.location
        }


@dataclass
class Memory:
    """
    A significant experience stored for future reference.
    Contains 14 fields as specified in the requirements.
    """
    
    id: str                              # Unique identifier
    timestamp: float                     # When it happened
    significance: float                  # 0.0-1.0: How important?
    emotional_impact: float              # -1.0 to +1.0: How did it feel?
    interaction_type: InteractionType    # Type of interaction
    participants: List[str]              # Who was involved?
    context: MemoryContext              # Context scores
    decay_factor: float                 # 0.0-1.0: Recency weight
    interaction_id: str                 # Link to original interaction
    outcome: str                        # "success"/"failure"/etc
    location: str                       # Where it happened
    context_tags: List[str]             # Tags like "betrayal", "victory"
    description: str                    # Human-readable description
    
    # Additional computed field
    weighted_significance: float = 0.0   # significance * decay_factor (computed)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.update_weighted_significance()
    
    def update_weighted_significance(self):
        """Update the weighted significance based on current decay."""
        self.weighted_significance = self.significance * self.decay_factor
    
    def apply_decay(self, time_elapsed_days: float):
        """
        Apply time-based decay to memory influence.
        
        Args:
            time_elapsed_days: Days since memory formation
        """
        # Exponential decay: decay = exp(-time / half_life)
        # Half-life varies by significance (important memories last longer)
        base_half_life = 30.0  # 30 days base
        significance_multiplier = 1.0 + self.significance * 2.0  # 1.0-3.0x
        half_life = base_half_life * significance_multiplier
        
        self.decay_factor = math.exp(-time_elapsed_days / half_life)
        self.update_weighted_significance()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'significance': self.significance,
            'emotional_impact': self.emotional_impact,
            'interaction_type': self.interaction_type.value,
            'participants': self.participants,
            'context': self.context.to_dict(),
            'decay_factor': self.decay_factor,
            'interaction_id': self.interaction_id,
            'outcome': self.outcome,
            'location': self.location,
            'context_tags': self.context_tags,
            'description': self.description,
            'weighted_significance': self.weighted_significance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        context_data = data['context']
        context = MemoryContext(
            goal_relevance=context_data['goal_relevance'],
            novelty_factor=context_data['novelty_factor'],
            social_importance=context_data['social_importance'],
            survival_relevance=context_data['survival_relevance'],
            participants=context_data['participants'],
            location=context_data['location']
        )
        
        memory = cls(
            id=data['id'],
            timestamp=data['timestamp'],
            significance=data['significance'],
            emotional_impact=data['emotional_impact'],
            interaction_type=InteractionType(data['interaction_type']),
            participants=data['participants'],
            context=context,
            decay_factor=data['decay_factor'],
            interaction_id=data['interaction_id'],
            outcome=data['outcome'],
            location=data['location'],
            context_tags=data['context_tags'],
            description=data['description']
        )
        
        memory.weighted_significance = data.get('weighted_significance', 0.0)
        return memory


class MemoryManager:
    """
    Manages memory formation, storage, retrieval, and pruning for agents.
    Implements the significance-based filtering and influence calculation.
    """
    
    MAX_MEMORIES_PER_AGENT = 50
    MIN_SIGNIFICANCE_THRESHOLD = 0.15  # Minimum significance to form memory
    
    def __init__(self):
        self.memories: List[Memory] = []
    
    def calculate_significance(self, event_data: Dict[str, Any], context: MemoryContext) -> float:
        """
        Calculate memory significance using the 5-component formula:
        significance = emotional_impact*0.4 + goal_relevance*0.3 + novelty*0.2 + social*0.1 + survival*0.1
        
        Args:
            event_data: Event details including emotional_impact
            context: Memory context with relevance scores
            
        Returns:
            Significance score 0.0-1.0
        """
        emotional_component = abs(event_data.get('emotional_impact', 0.0)) * 0.4
        goal_component = context.goal_relevance * 0.3
        novelty_component = context.novelty_factor * 0.2
        social_component = context.social_importance * 0.1
        survival_component = context.survival_relevance * 0.1
        
        total_significance = (emotional_component + goal_component + 
                            novelty_component + social_component + survival_component)
        
        return min(1.0, total_significance)
    
    def should_form_memory(self, significance: float) -> bool:
        """Check if event is significant enough to store as memory."""
        return significance >= self.MIN_SIGNIFICANCE_THRESHOLD
    
    def create_memory(self, event_data: Dict[str, Any], context: MemoryContext) -> Optional[Memory]:
        """
        Create a new memory if the event is significant enough.
        
        Args:
            event_data: Event details (emotional_impact, outcome, etc.)
            context: Memory context information
            
        Returns:
            Memory object if created, None if not significant enough
        """
        significance = self.calculate_significance(event_data, context)
        
        if not self.should_form_memory(significance):
            return None
        
        # Generate unique ID
        memory_id = f"mem_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"
        
        # Create memory
        memory = Memory(
            id=memory_id,
            timestamp=time.time(),
            significance=significance,
            emotional_impact=event_data.get('emotional_impact', 0.0),
            interaction_type=InteractionType(event_data.get('interaction_type', 'exploration')),
            participants=event_data.get('participants', []),
            context=context,
            decay_factor=1.0,  # Fresh memory
            interaction_id=event_data.get('interaction_id', ''),
            outcome=event_data.get('outcome', 'unknown'),
            location=event_data.get('location', 'unknown'),
            context_tags=event_data.get('context_tags', []),
            description=event_data.get('description', 'Significant event occurred')
        )
        
        # Add to memory storage
        self.add_memory(memory)
        return memory
    
    def add_memory(self, memory: Memory):
        """Add memory and enforce storage limits."""
        self.memories.append(memory)
        self._enforce_memory_limits()
    
    def _enforce_memory_limits(self):
        """Keep only the most significant memories (up to MAX_MEMORIES_PER_AGENT)."""
        if len(self.memories) <= self.MAX_MEMORIES_PER_AGENT:
            return
        
        # Update decay for all memories
        current_time = time.time()
        for memory in self.memories:
            days_elapsed = (current_time - memory.timestamp) / (24 * 3600)
            memory.apply_decay(days_elapsed)
        
        # Sort by weighted significance (desc) and keep top N
        self.memories.sort(key=lambda m: m.weighted_significance, reverse=True)
        self.memories = self.memories[:self.MAX_MEMORIES_PER_AGENT]
    
    def retrieve_relevant_memories(self, interaction_type: InteractionType, 
                                 max_count: int = 10) -> List[Memory]:
        """
        Retrieve relevant memories for decision making.
        
        Args:
            interaction_type: Type of interaction being considered
            max_count: Maximum number of memories to return
            
        Returns:
            List of relevant memories sorted by weighted significance
        """
        # Filter by interaction type
        relevant = [m for m in self.memories if m.interaction_type == interaction_type]
        
        # Update decay factors
        current_time = time.time()
        for memory in relevant:
            days_elapsed = (current_time - memory.timestamp) / (24 * 3600)
            memory.apply_decay(days_elapsed)
        
        # Sort by weighted significance (most impactful recent experiences first)
        relevant.sort(key=lambda m: m.weighted_significance, reverse=True)
        
        return relevant[:max_count]
    
    def calculate_memory_influence(self, interaction_type: InteractionType, 
                                   location: str = None, tags: List[str] = None) -> float:
        """
        Calculate memory influence multiplier for decision making.
        
        Args:
            interaction_type: Type of interaction being considered
            location: Optional location to filter memories
            tags: Optional tags to filter memories (e.g., direction, action type)
            
        Returns:
            Influence multiplier 0.8-1.5 (negative memories = avoid, positive = favor)
        """
        relevant_memories = self.retrieve_relevant_memories(interaction_type, 10)
        
        # Further filter by location or tags if provided
        if location:
            relevant_memories = [m for m in relevant_memories if m.location == location]
        
        if tags:
            relevant_memories = [m for m in relevant_memories 
                               if any(tag in m.context_tags for tag in tags)]
        
        if not relevant_memories:
            return 1.0  # Neutral influence (no relevant memories)
        
        # Calculate weighted emotional influence
        total_influence = 0.0
        total_weight = 0.0
        
        for memory in relevant_memories:
            weight = memory.weighted_significance
            influence = memory.emotional_impact * weight
            total_influence += influence
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        # Average influence weighted by significance
        average_influence = total_influence / total_weight
        
        # STRENGTHENED for maze learning: Map from [-1.0, 1.0] to [0.3, 2.5]
        # -1.0 → 0.3x (strong negative memory = STRONGLY avoid - was 0.8x)
        #  0.0 → 1.0x (neutral memories = no bias)
        # +1.0 → 2.5x (strong positive memory = STRONGLY favor - was 1.5x)
        multiplier = 1.0 + (average_influence * 1.5)
        
        return max(0.3, min(2.5, multiplier))
    
    def consolidate_similar_memories(self):
        """
        Quantum-inspired consolidation to merge similar memories for efficiency.
        Combines memories with similar outcomes and contexts.
        """
        if len(self.memories) < 5:
            return  # Not enough memories to consolidate
        
        # Group memories by interaction type and outcome
        groups = {}
        for memory in self.memories:
            key = (memory.interaction_type, memory.outcome)
            if key not in groups:
                groups[key] = []
            groups[key].append(memory)
        
        consolidated = []
        
        for group_memories in groups.values():
            if len(group_memories) <= 2:
                # Keep individual memories for small groups
                consolidated.extend(group_memories)
            else:
                # Consolidate into representative memory
                representative = self._create_representative_memory(group_memories)
                consolidated.append(representative)
        
        self.memories = consolidated
    
    def _create_representative_memory(self, memories: List[Memory]) -> Memory:
        """Create a single representative memory from a group of similar memories."""
        if not memories:
            raise ValueError("Cannot create representative from empty memory list")
        
        # Use the most significant memory as base
        base_memory = max(memories, key=lambda m: m.significance)
        
        # Average emotional impacts weighted by significance
        total_emotional = sum(m.emotional_impact * m.significance for m in memories)
        total_significance = sum(m.significance for m in memories)
        avg_emotional = total_emotional / total_significance if total_significance > 0 else 0.0
        
        # Use maximum significance (most impactful experience)
        max_significance = max(m.significance for m in memories)
        
        # Combine context tags
        all_tags = set()
        for memory in memories:
            all_tags.update(memory.context_tags)
        
        # Create consolidated memory
        representative = Memory(
            id=f"consolidated_{base_memory.id}",
            timestamp=base_memory.timestamp,  # Use timestamp of most significant
            significance=max_significance,
            emotional_impact=avg_emotional,
            interaction_type=base_memory.interaction_type,
            participants=base_memory.participants,
            context=base_memory.context,
            decay_factor=base_memory.decay_factor,
            interaction_id=base_memory.interaction_id,
            outcome=base_memory.outcome,
            location=base_memory.location,
            context_tags=list(all_tags),
            description=f"Consolidated memory from {len(memories)} similar experiences"
        )
        
        return representative
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state."""
        if not self.memories:
            return {'total_memories': 0}
        
        # Update decay factors
        current_time = time.time()
        for memory in self.memories:
            days_elapsed = (current_time - memory.timestamp) / (24 * 3600)
            memory.apply_decay(days_elapsed)
        
        by_type = {}
        by_outcome = {}
        
        for memory in self.memories:
            # Count by type
            type_key = memory.interaction_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Count by outcome
            by_outcome[memory.outcome] = by_outcome.get(memory.outcome, 0) + 1
        
        return {
            'total_memories': len(self.memories),
            'avg_significance': sum(m.significance for m in self.memories) / len(self.memories),
            'avg_decay': sum(m.decay_factor for m in self.memories) / len(self.memories),
            'by_type': by_type,
            'by_outcome': by_outcome,
            'emotional_range': {
                'min': min(m.emotional_impact for m in self.memories),
                'max': max(m.emotional_impact for m in self.memories),
                'avg': sum(m.emotional_impact for m in self.memories) / len(self.memories)
            }
        }
    
    def clear_memories(self):
        """Clear all memories (for testing/reset)."""
        self.memories.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire memory system to dictionary."""
        return {
            'memories': [m.to_dict() for m in self.memories],
            'stats': self.get_memory_stats()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load memory system from dictionary."""
        self.memories = [Memory.from_dict(m) for m in data['memories']]