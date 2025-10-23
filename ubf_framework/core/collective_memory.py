"""
Universal Behavioral Framework - Collective Memory

Shared memory pool that allows agents to learn from each other's experiences.
When an agent succeeds, it broadcasts key memories to help other agents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .memory_system import Memory, InteractionType


@dataclass
class CollectiveMemory:
    """
    A memory shared by the collective that originated from a successful agent.
    Contains attribution and strength based on how many agents contributed similar experiences.
    """
    base_memory: Memory  # The original memory
    contributor_count: int = 1  # How many agents contributed to this pattern
    success_count: int = 0  # How many times this led to success
    failure_count: int = 0  # How many times this pattern failed
    
    @property
    def reliability(self) -> float:
        """Calculate reliability score (0.0-1.0) based on success/failure ratio."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral for untested patterns
        return self.success_count / total
    
    @property
    def collective_significance(self) -> float:
        """
        Enhanced significance based on collective validation.
        More contributors = stronger signal.
        """
        base_sig = self.base_memory.weighted_significance
        contributor_boost = 1.0 + (self.contributor_count * 0.1)  # +10% per contributor
        reliability_mult = 0.5 + (self.reliability * 0.5)  # 0.5x-1.0x based on reliability
        return base_sig * contributor_boost * reliability_mult


class CollectiveMemoryPool:
    """
    Manages shared memories across all agents in a group.
    Agents contribute memories when successful, query when making decisions.
    """
    
    def __init__(self):
        self.memories: List[CollectiveMemory] = []
        self.contribution_log: List[Dict[str, Any]] = []  # Track who contributed what
        
    def broadcast_success_memories(self, agent_id: str, memories: List[Memory], 
                                   context: Optional[Dict[str, Any]] = None):
        """
        Broadcast memories from a successful agent to the collective pool.
        
        Args:
            agent_id: ID of the contributing agent
            memories: List of memories to share (typically the most significant ones)
            context: Additional context about the success (final position, steps taken, etc.)
        """
        contributed_count = 0
        
        for memory in memories:
            # Only share memories with reasonable significance
            if memory.weighted_significance < 0.1:
                continue
                
            # Check if similar memory already exists
            existing = self._find_similar_memory(memory)
            
            if existing:
                # Strengthen existing pattern
                existing.contributor_count += 1
                if memory.outcome == 'success' or memory.outcome == 'goal_achieved':
                    existing.success_count += 1
                elif memory.outcome in ['failure', 'collision']:
                    existing.failure_count += 1
            else:
                # Add new collective memory
                collective_mem = CollectiveMemory(
                    base_memory=memory,
                    contributor_count=1,
                    success_count=1 if memory.outcome in ['success', 'goal_achieved'] else 0,
                    failure_count=1 if memory.outcome in ['failure', 'collision'] else 0
                )
                self.memories.append(collective_mem)
            
            contributed_count += 1
        
        # Log the contribution
        self.contribution_log.append({
            'agent_id': agent_id,
            'memory_count': contributed_count,
            'context': context or {}
        })
        
        return contributed_count
    
    def query_collective_knowledge(self, interaction_type: InteractionType,
                                   location: str = None, 
                                   tags: List[str] = None,
                                   min_reliability: float = 0.3) -> List[CollectiveMemory]:
        """
        Query collective memories relevant to current situation.
        
        Args:
            interaction_type: Type of interaction being considered
            location: Optional location filter
            tags: Optional tag filters (direction, action, etc.)
            min_reliability: Minimum reliability threshold (0.0-1.0)
            
        Returns:
            List of relevant collective memories, sorted by significance
        """
        relevant = []
        
        for col_mem in self.memories:
            memory = col_mem.base_memory
            
            # Filter by interaction type
            if memory.interaction_type != interaction_type:
                continue
            
            # Filter by location if specified
            if location and memory.location != location:
                continue
            
            # Filter by tags if specified
            if tags:
                if not any(tag in memory.context_tags for tag in tags):
                    continue
            
            # Filter by reliability
            if col_mem.reliability < min_reliability:
                continue
            
            relevant.append(col_mem)
        
        # Sort by collective significance (highest first)
        relevant.sort(key=lambda x: x.collective_significance, reverse=True)
        
        return relevant
    
    def calculate_collective_influence(self, interaction_type: InteractionType,
                                      location: str = None,
                                      tags: List[str] = None) -> float:
        """
        Calculate influence multiplier from collective memories (similar to personal memory influence).
        
        Returns:
            Multiplier 0.8-1.5 based on collective wisdom
        """
        relevant = self.query_collective_knowledge(interaction_type, location, tags)
        
        if not relevant:
            return 1.0  # Neutral (no collective knowledge)
        
        # Calculate weighted influence from collective memories
        total_influence = 0.0
        total_weight = 0.0
        
        for col_mem in relevant:
            weight = col_mem.collective_significance
            # Use emotional impact from base memory, but scale by reliability
            influence = col_mem.base_memory.emotional_impact * col_mem.reliability * weight
            total_influence += influence
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        average_influence = total_influence / total_weight
        
        # Map from [-1.0, 1.0] to [0.8, 1.5]
        # But make collective influence slightly weaker than personal memory
        multiplier = 1.0 + (average_influence * 0.4)  # 0.4 instead of 0.5 for personal
        
        return max(0.8, min(1.5, multiplier))
    
    def _find_similar_memory(self, memory: Memory) -> Optional[CollectiveMemory]:
        """
        Find existing collective memory with similar pattern.
        Similar = same location, same interaction type, overlapping tags.
        """
        for col_mem in self.memories:
            existing = col_mem.base_memory
            
            # Same location and interaction type?
            if (existing.location == memory.location and 
                existing.interaction_type == memory.interaction_type):
                
                # Check for significant tag overlap
                common_tags = set(existing.context_tags) & set(memory.context_tags)
                
                # If they share direction or action tags, consider them similar
                relevant_tags = [tag for tag in common_tags 
                               if tag.startswith('dir_') or tag.startswith('action_')]
                
                if relevant_tags:
                    return col_mem
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collective memory pool."""
        if not self.memories:
            return {
                'total_memories': 0,
                'contributors': 0,
                'avg_reliability': 0.0,
                'most_reliable_pattern': None
            }
        
        total_contributors = sum(cm.contributor_count for cm in self.memories)
        avg_reliability = sum(cm.reliability for cm in self.memories) / len(self.memories)
        
        # Find most reliable pattern
        most_reliable = max(self.memories, key=lambda x: x.reliability * x.contributor_count)
        
        return {
            'total_memories': len(self.memories),
            'contributors': total_contributors,
            'avg_reliability': avg_reliability,
            'most_reliable_pattern': {
                'location': most_reliable.base_memory.location,
                'tags': most_reliable.base_memory.context_tags,
                'reliability': most_reliable.reliability,
                'contributors': most_reliable.contributor_count
            }
        }
    
    def clear(self):
        """Clear all collective memories (for fresh starts)."""
        self.memories.clear()
        self.contribution_log.clear()
