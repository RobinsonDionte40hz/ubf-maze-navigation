"""
Universal Behavioral Framework - Consciousness Update Service

Implements event-driven coordinate updates with enhanced failure learning,
noise injection, and creative adaptation mechanisms.
"""

import time
import math
import random
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .consciousness_state import ConsciousnessState, BehavioralState
from .memory_system import MemoryManager, MemoryContext, InteractionType


class EventType(Enum):
    """Types of events that can trigger consciousness updates."""
    GOAL_COMPLETION = "goal_completion"
    GOAL_FAILURE = "goal_failure"
    TRAUMATIC_ENCOUNTER = "traumatic_encounter"
    CONFLICT_RESOLUTION = "conflict_resolution"
    SOCIAL_SUCCESS = "social_success"
    SOCIAL_REJECTION = "social_rejection"
    LEARNING_SUCCESS = "learning_success"
    EXPLORATION_SUCCESS = "exploration_success"
    EXPLORATION_FAILURE = "exploration_failure"
    INSIGHTFUL_FAILURE = "insightful_failure"  # Enhanced failure type
    CREATIVE_BREAKTHROUGH = "creative_breakthrough"
    SURVIVAL_SUCCESS = "survival_success"
    SURVIVAL_FAILURE = "survival_failure"
    ECONOMIC_SUCCESS = "economic_success"
    ECONOMIC_FAILURE = "economic_failure"
    ROUTINE_SUCCESS = "routine_success"
    MINOR_SETBACK = "minor_setback"


@dataclass
class EventData:
    """Data structure for events that trigger consciousness updates."""
    event_type: EventType
    emotional_impact: float  # -1.0 to +1.0
    outcome: str            # "success", "failure", "mixed", etc.
    significance: float     # 0.0-1.0 (calculated or provided)
    interaction_type: InteractionType
    participants: list = None
    location: str = ""
    context_tags: list = None
    description: str = ""
    novelty_factor: float = 0.0
    goal_relevance: float = 0.5
    social_importance: float = 0.0
    survival_relevance: float = 0.0
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.context_tags is None:
            self.context_tags = []


class ConsciousnessUpdateService:
    """
    Manages consciousness coordinate updates based on experience outcomes.
    Implements enhanced failure learning and creative adaptation.
    """
    
    # Event type to coordinate change mappings
    UPDATE_RULES = {
        EventType.GOAL_COMPLETION: {
            'frequency': +0.3,
            'coherence': +0.05
        },
        EventType.GOAL_FAILURE: {
            'frequency': -0.2,  # REDUCED from -0.3: bruised, not paralyzed
            'coherence': +0.05  # Balanced learning - failures provide focus
        },
        EventType.TRAUMATIC_ENCOUNTER: {
            'frequency': -1.0,
            'coherence': -0.2
        },
        EventType.CONFLICT_RESOLUTION: {
            'frequency': +0.2,
            'coherence': +0.02  # Victory case
        },
        EventType.SOCIAL_SUCCESS: {
            'frequency': +0.2,
            'coherence': +0.03
        },
        EventType.SOCIAL_REJECTION: {
            'frequency': -0.2,  # REDUCED from -0.4: softer penalty
            'coherence': -0.05
        },
        EventType.LEARNING_SUCCESS: {
            'frequency': +0.1,
            'coherence': +0.1
        },
        EventType.EXPLORATION_SUCCESS: {
            'frequency': +0.2,
            'coherence': +0.02
        },
        EventType.EXPLORATION_FAILURE: {
            'frequency': -0.2,
            'coherence': +0.03  # Learning from exploration failure
        },
        EventType.INSIGHTFUL_FAILURE: {
            'frequency': +0.1,
            'coherence': +0.1   # Enhanced learning from creative failure
        },
        EventType.CREATIVE_BREAKTHROUGH: {
            'frequency': +0.4,
            'coherence': +0.08
        },
        EventType.SURVIVAL_SUCCESS: {
            'frequency': +0.3,
            'coherence': +0.05
        },
        EventType.SURVIVAL_FAILURE: {
            'frequency': -0.8,
            'coherence': -0.15
        },
        EventType.ECONOMIC_SUCCESS: {
            'frequency': +0.15,
            'coherence': +0.03
        },
        EventType.ECONOMIC_FAILURE: {
            'frequency': -0.25,
            'coherence': +0.02  # Learning from economic mistakes
        },
        EventType.ROUTINE_SUCCESS: {
            'frequency': +0.05,
            'coherence': +0.01
        },
        EventType.MINOR_SETBACK: {
            'frequency': -0.1,
            'coherence': +0.01  # Small learning opportunity
        }
    }
    
    SIGNIFICANCE_THRESHOLD = 0.15  # Lowered to capture more learning events
    
    def __init__(self, noise_scheduler: Optional['NoiseScheduler'] = None):
        self.noise_scheduler = noise_scheduler or NoiseScheduler()
        self.last_update_time = time.time()
    
    def process_event(self, 
                     consciousness: ConsciousnessState,
                     memory_manager: MemoryManager,
                     event_data: EventData) -> Tuple[bool, Optional[BehavioralState]]:
        
        # Create memory context
        context = MemoryContext(
            goal_relevance=event_data.goal_relevance,
            novelty_factor=event_data.novelty_factor,
            social_importance=event_data.social_importance,
            survival_relevance=event_data.survival_relevance,
            participants=event_data.participants,
            location=event_data.location
        )
        
        # Calculate significance if not provided
        if event_data.significance == 0.0:
            event_dict = {
                'emotional_impact': event_data.emotional_impact,
                'interaction_type': event_data.interaction_type.value,
                'outcome': event_data.outcome,
                'participants': event_data.participants,
                'context_tags': event_data.context_tags,
                'description': event_data.description
            }
            event_data.significance = memory_manager.calculate_significance(event_dict, context)
        
        # Check if event is significant enough to update consciousness
        if event_data.significance < self.SIGNIFICANCE_THRESHOLD:
            return False, None
        
        # Determine if this is an insightful failure
        event_data = self._check_for_insightful_failure(event_data)
        
        # Get coordinate changes for this event type
        freq_delta, coh_delta = self._get_coordinate_changes(event_data)
        
        # Apply magnitude scaling based on significance and emotional impact
        magnitude_scale = event_data.significance * (0.5 + abs(event_data.emotional_impact) * 0.5)
        freq_delta *= magnitude_scale
        coh_delta *= magnitude_scale
        
        # Get current noise level from scheduler
        noise_std = self.noise_scheduler.get_current_noise(
            consciousness=consciousness,
            recent_failure=(event_data.outcome == "failure")
        )
        
        # Update consciousness coordinates with noise
        consciousness.update_coordinates(freq_delta, coh_delta, noise_std)
        consciousness.last_updated = time.time()
        
        # Generate new behavioral state
        new_behavioral_state = BehavioralState.from_consciousness(consciousness)
        
        # Create memory of this significant event
        memory_event = {
            'emotional_impact': event_data.emotional_impact,
            'interaction_type': event_data.interaction_type.value,
            'outcome': event_data.outcome,
            'participants': event_data.participants,
            'context_tags': event_data.context_tags,
            'description': event_data.description,
            'location': event_data.location,
            'interaction_id': f"event_{int(time.time() * 1000000)}"
        }
        
        memory_manager.create_memory(memory_event, context)
        
        return True, new_behavioral_state
    
    def _check_for_insightful_failure(self, event_data: EventData) -> EventData:
        """
        Check if a failure should be converted to an insightful failure.
        
        Criteria: failure + high novelty (>0.3) + high significance (>0.5)
        """
        if (event_data.outcome == "failure" and 
            event_data.novelty_factor > 0.3 and 
            event_data.significance > 0.5):
            
            # Convert to insightful failure
            event_data.event_type = EventType.INSIGHTFUL_FAILURE
            event_data.context_tags.append("insightful")
            event_data.description += " (converted to insightful failure)"
            
            # Boost emotional impact slightly (learning is rewarding)
            event_data.emotional_impact = max(-0.5, event_data.emotional_impact + 0.2)
        
        return event_data
    
    def _get_coordinate_changes(self, event_data: EventData) -> Tuple[float, float]:
        """
        Get frequency and coherence changes for an event.
        
        Args:
            event_data: Event data
            
        Returns:
            Tuple of (frequency_delta, coherence_delta)
        """
        base_changes = self.UPDATE_RULES.get(event_data.event_type, {'frequency': 0.0, 'coherence': 0.0})
        freq_delta = base_changes['frequency']
        coh_delta = base_changes['coherence']
        
        # Special handling for conflict resolution based on outcome
        if event_data.event_type == EventType.CONFLICT_RESOLUTION:
            if event_data.outcome == "victory":
                freq_delta = +0.2
                coh_delta = +0.02
            elif event_data.outcome == "defeat":
                freq_delta = -0.3
                coh_delta = -0.05
            else:  # draw/mixed
                freq_delta = -0.1
                coh_delta = +0.01
        
        return freq_delta, coh_delta
    
    def create_event_from_action_outcome(self,
                                       action_type: str,
                                       outcome: str,
                                       interaction_type: InteractionType,
                                       context: Dict[str, Any]) -> EventData:
        """
        Create an EventData object from an action outcome.
        
        Args:
            action_type: Type of action taken
            outcome: Result of the action ("success", "failure", etc.)
            interaction_type: Type of interaction
            context: Additional context information
            
        Returns:
            EventData object ready for processing
        """
        # Map action outcomes to event types
        if outcome == "success":
            if interaction_type == InteractionType.EXPLORATION:
                event_type = EventType.EXPLORATION_SUCCESS
            elif interaction_type == InteractionType.SOCIAL:
                event_type = EventType.SOCIAL_SUCCESS
            elif interaction_type == InteractionType.LEARNING:
                event_type = EventType.LEARNING_SUCCESS
            elif interaction_type == InteractionType.SURVIVAL:
                event_type = EventType.SURVIVAL_SUCCESS
            elif interaction_type == InteractionType.ECONOMIC:
                event_type = EventType.ECONOMIC_SUCCESS
            elif interaction_type == InteractionType.CREATIVE:
                event_type = EventType.CREATIVE_BREAKTHROUGH
            else:
                event_type = EventType.ROUTINE_SUCCESS
        
        elif outcome == "failure" or outcome == "collision":
            if interaction_type == InteractionType.EXPLORATION:
                event_type = EventType.EXPLORATION_FAILURE
            elif interaction_type == InteractionType.SOCIAL:
                event_type = EventType.SOCIAL_REJECTION
            elif interaction_type == InteractionType.SURVIVAL:
                event_type = EventType.SURVIVAL_FAILURE
            elif interaction_type == InteractionType.ECONOMIC:
                event_type = EventType.ECONOMIC_FAILURE
            else:
                event_type = EventType.GOAL_FAILURE
        
        else:  # partial success, mixed, etc.
            event_type = EventType.MINOR_SETBACK
        
        # Determine emotional impact based on outcome and context
        emotional_impact = self._calculate_emotional_impact(outcome, context)
        
        return EventData(
            event_type=event_type,
            emotional_impact=emotional_impact,
            outcome=outcome,
            significance=0.0,  # Will be calculated in process_event
            interaction_type=interaction_type,
            participants=context.get('participants', []),
            location=context.get('location', ''),
            context_tags=context.get('tags', []),
            description=context.get('description', f"{action_type} resulted in {outcome}"),
            novelty_factor=context.get('novelty_factor', 0.0),
            goal_relevance=context.get('goal_relevance', 0.5),
            social_importance=context.get('social_importance', 0.0),
            survival_relevance=context.get('survival_relevance', 0.0)
        )
    
    def _calculate_emotional_impact(self, outcome: str, context: Dict[str, Any]) -> float:
        """Calculate emotional impact based on outcome and context."""
        base_impact = {
            "success": +0.6,
            "failure": -0.6,
            "collision": -0.8,  # Collisions are bad! Strong negative
            "goal_achieved": +1.0,  # Reaching goal is amazing!
            "victory": +0.8,
            "defeat": -0.8,
            "mixed": +0.1,
            "partial": +0.3,
            "critical_success": +0.9,
            "critical_failure": -0.9
        }.get(outcome, 0.0)
        
        # Modify based on context
        importance = context.get('importance', 0.5)
        surprise = context.get('surprise_factor', 0.0)
        
        # Important events have stronger emotional impact
        impact_modifier = 0.5 + importance * 0.5
        
        # Surprising events have stronger emotional impact
        surprise_modifier = 1.0 + surprise * 0.3
        
        final_impact = base_impact * impact_modifier * surprise_modifier
        
        return max(-1.0, min(1.0, final_impact))


class NoiseScheduler:
    """
    Manages noise injection for creativity and exploration.
    Increases noise after failures and during low coherence states.
    """
    
    def __init__(self, base_noise: float = 0.1, max_noise: float = 0.2):
        self.base_noise = base_noise
        self.max_noise = max_noise
        self.failure_boost_factor = 2.0
        self.coherence_amplification = True
        
        # COHERENCE RECOVERY MECHANISM
        self.low_coherence_threshold = 0.4  # Trigger recovery below this
        self.recovery_mode = False
        self.recovery_steps_remaining = 0
        self.recovery_steps_total = 5  # Five steps of high exploration
        self.recovery_temperature = 1.8  # Higher temperature for bold moves
        self.recovery_noise = 0.2  # Higher noise for exploration
    
    def get_current_noise(self, consciousness: ConsciousnessState, recent_failure: bool = False) -> float:
        """
        Calculate current noise level based on state and recent events.
        
        Args:
            consciousness: Current consciousness state
            recent_failure: Whether agent recently experienced failure
            
        Returns:
            Current noise standard deviation
        """
        # CHECK FOR LOW COHERENCE RECOVERY MODE
        if consciousness.coherence < self.low_coherence_threshold and not self.recovery_mode:
            # Activate recovery mode!
            self.recovery_mode = True
            self.recovery_steps_remaining = self.recovery_steps_total
        
        # IN RECOVERY MODE: inject high noise for exploration
        if self.recovery_mode:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.recovery_mode = False  # Exit recovery
            return self.recovery_noise  # High noise to break out of rut
        
        # NORMAL OPERATION
        noise = self.base_noise
        
        # Amplify noise for low coherence (scattered = more creative)
        if self.coherence_amplification:
            coherence_factor = max(0.5, 1.0 - consciousness.coherence)
            noise *= coherence_factor
        
        # Boost noise after failures for serendipitous discovery
        if recent_failure:
            noise *= self.failure_boost_factor
        
        return min(self.max_noise, noise)
    
    def get_temperature(self) -> float:
        """Get current decision temperature (higher in recovery mode)."""
        return self.recovery_temperature if self.recovery_mode else 1.0
    
    def anneal_noise(self, step: int, total_steps: int):
        """
        Anneal noise over time (optional for long-term learning).
        
        Args:
            step: Current step
            total_steps: Total steps in episode
        """
        progress = step / total_steps
        # Reduce noise by up to 50% over time
        self.base_noise = self.base_noise * (1.0 - 0.5 * progress)
    
    def reset_noise(self):
        """Reset noise to initial values."""
        self.base_noise = 0.1
        self.max_noise = 0.2


# Utility functions for creating common event types
def create_maze_exploration_event(success: bool, novelty: float, location: str = "") -> EventData:
    """Create event for maze exploration outcome."""
    outcome = "success" if success else "failure"
    emotional_impact = +0.4 if success else -0.3
    
    return EventData(
        event_type=EventType.EXPLORATION_SUCCESS if success else EventType.EXPLORATION_FAILURE,
        emotional_impact=emotional_impact,
        outcome=outcome,
        significance=0.0,  # Will be calculated
        interaction_type=InteractionType.EXPLORATION,
        location=location,
        context_tags=["maze", "navigation"],
        description=f"Maze exploration {outcome}",
        novelty_factor=novelty,
        goal_relevance=0.8,  # High relevance for maze solving
        survival_relevance=0.1
    )


def create_goal_completion_event(goal_achieved: bool, importance: float = 0.8) -> EventData:
    """Create event for goal completion/failure."""
    if goal_achieved:
        return EventData(
            event_type=EventType.GOAL_COMPLETION,
            emotional_impact=+0.8 * importance,
            outcome="success",
            significance=0.0,
            interaction_type=InteractionType.LEARNING,
            context_tags=["goal", "achievement"],
            description="Major goal achieved",
            goal_relevance=1.0,
            novelty_factor=0.3
        )
    else:
        return EventData(
            event_type=EventType.GOAL_FAILURE,
            emotional_impact=-0.6 * importance,
            outcome="failure", 
            significance=0.0,
            interaction_type=InteractionType.LEARNING,
            context_tags=["goal", "failure"],
            description="Failed to achieve goal",
            goal_relevance=1.0,
            novelty_factor=0.2
        )