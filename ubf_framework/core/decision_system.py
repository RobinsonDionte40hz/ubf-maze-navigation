"""
Universal Behavioral Framework - Decision System

Implements the 13-factor decision weighting system with resonance-based
action selection and temperature-based exploration.
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from .consciousness_state import ConsciousnessState, BehavioralState
from .memory_system import MemoryManager, InteractionType

if TYPE_CHECKING:
    from .collective_memory import CollectiveMemoryPool


class ActionType(Enum):
    """Types of actions available in the system."""
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    EXPLORE = "explore"
    INVESTIGATE = "investigate"
    REST = "rest"
    SOCIALIZE = "socialize"
    AVOID = "avoid"
    BACKTRACK = "backtrack"
    CREATIVE_SOLUTION = "creative_solution"


@dataclass
class Action:
    """Represents a possible action the agent can take."""
    action_type: ActionType
    interaction_type: InteractionType
    base_weight: float = 1.0
    energy_requirement: float = 0.1  # 0.0-1.0
    focus_requirement: float = 0.1   # 0.0-1.0
    risk_level: float = 0.1          # 0.0-1.0
    social_component: float = 0.0    # 0.0-1.0
    creativity_component: float = 0.0 # 0.0-1.0
    goal_alignment: float = 0.5      # 0.0-1.0 (how well this action serves goals)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.value,
            'interaction_type': self.interaction_type.value,
            'base_weight': self.base_weight,
            'energy_requirement': self.energy_requirement,
            'focus_requirement': self.focus_requirement,
            'risk_level': self.risk_level,
            'social_component': self.social_component,
            'creativity_component': self.creativity_component,
            'goal_alignment': self.goal_alignment,
            'description': self.description
        }


class DecisionSystem:
    """
    Implements the 13-factor decision weighting system with quantum resonance
    and temperature-based probabilistic selection.
    """
    
    def __init__(self, temperature: float = 1.0, noise_std: float = 0.1):
        self.temperature = temperature  # 0.1-2.0 for exploration control
        self.noise_std = noise_std     # Gaussian noise standard deviation
        self.gamma_freq = 40.0         # 40 Hz gamma baseline for resonance
        
        # Temperature scheduling
        self.base_temperature = temperature
        self.failure_temp_boost = 0.0
        self.failure_steps_remaining = 0
    
    def calculate_interaction_weight(self, 
                                   consciousness: ConsciousnessState,
                                   behavioral_state: BehavioralState,
                                   memory_manager: MemoryManager,
                                   action: Action,
                                   environmental_context: Dict[str, Any] = None,
                                   collective_memory: 'CollectiveMemoryPool' = None) -> float:
        """
        Calculate the complete interaction weight using all 13 factors.
        
        Args:
            consciousness: Current consciousness coordinates
            behavioral_state: Cached behavioral state
            memory_manager: Memory system for influence calculation
            action: Action being evaluated
            environmental_context: Current environment state
            collective_memory: Optional collective memory pool for group learning
            
        Returns:
            Final weighted score for this action
        """
        if environmental_context is None:
            environmental_context = {}
        
        weight = action.base_weight
        
        # Factor 1: Goal alignment (+10.0 dominant boost)
        goal_factor = self._calculate_goal_priority(action, environmental_context)
        weight += goal_factor
        
        # Factor 2: Critical needs (+8.0 for survival)
        needs_factor = self._calculate_critical_needs(behavioral_state, action)
        weight += needs_factor
        
        # Factor 3: Environmental suitability (0.1x-3.0x)
        env_factor = self._calculate_environmental_suitability(action, environmental_context)
        weight *= env_factor
        
        # Factor 4: Personality influence (+2.0)
        personality_factor = self._calculate_personality_influence(behavioral_state, action)
        weight += personality_factor
        
        # Factor 5: MEMORY INFLUENCE (0.3x-2.5x multiplier) - CRITICAL FOR LEARNING!
        # For MOVE_FORWARD, check memories at the TARGET location + direction
        # For TURNS, check what direction from current location worked best
        current_location = environmental_context.get('current_location', None)
        current_direction = environmental_context.get('current_direction', None)
        
        memory_tags = [f"action_{action.action_type.value}"]
        check_location = current_location  # Default: check current location
        
        if action.action_type == ActionType.MOVE_FORWARD and current_direction is not None:
            # Calculate where we WOULD move to
            # Parse current position from location string "pos_(x, y)"
            if current_location and current_location.startswith("pos_"):
                try:
                    pos_str = current_location[4:]  # Remove "pos_"
                    pos = eval(pos_str)  # Convert "(x, y)" to tuple
                    x, y = pos
                    
                    # Calculate next position based on direction
                    dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
                    dx, dy = dir_vectors[current_direction]
                    next_pos = (x + dx, y + dy)
                    check_location = f"pos_{next_pos}"  # Check TARGET location's memories!
                    memory_tags.append(f"dir_{current_direction}")
                except:
                    pass  # If parsing fails, use current location
        
        elif action.action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
            # For turns, check memories from current location in the NEW direction
            if current_location and current_location.startswith("pos_") and current_direction is not None:
                try:
                    # Calculate what direction we'll face after turning
                    if action.action_type == ActionType.TURN_LEFT:
                        new_direction = (current_direction - 1) % 4
                    else:  # TURN_RIGHT
                        new_direction = (current_direction + 1) % 4
                    
                    # Check memories of moving forward in that direction from current spot
                    memory_tags.append(f"dir_{new_direction}")
                    memory_tags.append("action_MOVE_FORWARD")  # What happened when we moved in that direction?
                except:
                    pass
        
        # Personal memory influence at TARGET location
        memory_factor = memory_manager.calculate_memory_influence(
            action.interaction_type, 
            location=check_location,
            tags=memory_tags if memory_tags else None
        )
        weight *= memory_factor
        
        # COLLECTIVE MEMORY INFLUENCE (Factor 5b: group learning)
        # If collective memory available, also consider what the group has learned
        if collective_memory is not None:
            collective_factor = collective_memory.calculate_collective_influence(
                action.interaction_type,
                location=check_location,  # Also use target location for collective
                tags=memory_tags if memory_tags else None
            )
            # Weight collective memories slightly less than personal (0.7x strength)
            weight *= (1.0 + (collective_factor - 1.0) * 0.7)
        
        # Factor 6: Emotional state (0.1x-3.0x)
        emotional_factor = self._calculate_emotional_influence(behavioral_state, action)
        weight *= emotional_factor
        
        # Factor 7: Resource constraints (0.0x-1.0x)
        resource_factor = self._calculate_resource_constraints(behavioral_state, action)
        weight *= resource_factor
        
        # Factor 8: Social dynamics (+1.5)
        social_factor = self._calculate_social_dynamics(behavioral_state, action, environmental_context)
        weight += social_factor
        
        # Factor 9: Learning opportunities (+1.0)
        learning_factor = self._calculate_learning_opportunities(action, environmental_context)
        weight += learning_factor
        
        # Factor 10: Risk assessment (-2.0 to +2.0)
        risk_factor = self._calculate_risk_assessment(behavioral_state, action)
        weight += risk_factor
        
        # Factor 11: Consciousness modifier (behavioral state influence)
        consciousness_factor = self._calculate_consciousness_modifier(
            consciousness, behavioral_state, action)
        weight *= consciousness_factor
        
        # Factor 12: Temporal factors (+0.5)
        temporal_factor = self._calculate_temporal_factors(action, environmental_context)
        weight += temporal_factor
        
        # Factor 13: Quantum resonance (quantum coherence-based)
        resonance_factor = self._calculate_quantum_resonance(consciousness, action)
        weight *= resonance_factor
        
        # Add Gaussian noise for creativity
        noise = random.gauss(0, self.noise_std)
        weight += noise
        
        return max(0.01, weight)  # Ensure positive weight
    
    def _calculate_goal_priority(self, action: Action, context: Dict[str, Any]) -> float:
        """Factor 1: Goal alignment (dominant boost up to +10.0)."""
        goal_progress = context.get('goal_progress', 0.5)
        goal_urgency = context.get('goal_urgency', 0.5)
        moving_toward_goal = context.get('moving_toward_goal', False)
        
        # Strong bonus for movement toward goal
        if action.action_type == ActionType.MOVE_FORWARD and moving_toward_goal:
            return 12.0  # Extra boost for goal-directed movement
        
        # High alignment with urgent goals gets maximum boost
        alignment_score = action.goal_alignment * goal_urgency
        return alignment_score * 10.0
    
    def _calculate_critical_needs(self, behavioral_state: BehavioralState, action: Action) -> float:
        """Factor 2: Critical needs (+8.0 for survival)."""
        # Simulate critical needs (energy depletion, danger, etc.)
        energy_critical = 1.0 - behavioral_state.energy  # More critical when low energy
        
        if action.action_type == ActionType.REST and energy_critical > 0.7:
            return 8.0 * energy_critical
        elif action.action_type == ActionType.AVOID and action.risk_level < 0.3:
            return 6.0 * energy_critical
        
        return 0.0
    
    def _calculate_environmental_suitability(self, action: Action, context: Dict[str, Any]) -> float:
        """Factor 3: Environmental suitability (0.1x-3.0x multiplier)."""
        # Check if environment supports this action
        terrain_difficulty = context.get('terrain_difficulty', 0.5)
        visibility = context.get('visibility', 1.0)
        obstacles = context.get('obstacles_nearby', 0.0)
        moving_toward_goal = context.get('moving_toward_goal', False)
        
        if action.action_type in [ActionType.MOVE_FORWARD, ActionType.EXPLORE]:
            # Movement toward goal gets bonus even with obstacles
            if action.action_type == ActionType.MOVE_FORWARD and moving_toward_goal:
                return 2.5  # Strong boost for goal-directed movement
            
            # Movement actions less affected by obstacles (maze context)
            suitability = (1.0 - terrain_difficulty * 0.3) * visibility * (1.0 - obstacles * 0.3)
            return max(0.5, min(3.0, 0.5 + suitability * 2.5))
        elif action.action_type in [ActionType.INVESTIGATE, ActionType.CREATIVE_SOLUTION]:
            # Investigation benefits from good visibility
            return max(0.1, min(3.0, 0.5 + visibility * 2.5))
        else:
            return 1.0  # Neutral for other actions
    
    def _calculate_personality_influence(self, behavioral_state: BehavioralState, action: Action) -> float:
        """Factor 4: Personality influence (+2.0). SIMPLIFIED FOR MAZE - just risk tolerance."""
        personality_match = 0.0
        
        # Only risk tolerance matters for maze navigation
        if action.risk_level > 0.5:
            personality_match += behavioral_state.risk_tolerance
        
        if action.goal_alignment > 0.7:
            personality_match += behavioral_state.ambition
        
        return personality_match * 2.0
    
    def _calculate_emotional_influence(self, behavioral_state: BehavioralState, action: Action) -> float:
        """Factor 6: Emotional state (0.1x-3.0x multiplier). SIMPLIFIED FOR MAZE - basic mood only."""
        mood = behavioral_state.mood
        
        # Positive mood favors forward movement, negative mood neutral
        if action.action_type == ActionType.EXPLORE:
            if mood > 0:
                return 1.0 + mood * 2.0  # Up to 3.0x for very positive mood
            else:
                return max(0.1, 1.0 + mood * 0.9)  # Down to 0.1x for very negative mood
        
        # Negative mood favors defensive actions
        elif action.action_type in [ActionType.REST, ActionType.AVOID, ActionType.BACKTRACK]:
            if mood < 0:
                return 1.0 + abs(mood) * 2.0  # Up to 3.0x for very negative mood
            else:
                return max(0.1, 1.0 - mood * 0.9)  # Down to 0.1x for very positive mood
        
        return 1.0  # Neutral for other actions
    
    def _calculate_resource_constraints(self, behavioral_state: BehavioralState, action: Action) -> float:
        """Factor 7: Resource constraints (0.0x-1.0x multiplier)."""
        # Check if agent has enough energy/focus for action
        energy_available = behavioral_state.energy >= action.energy_requirement
        focus_available = behavioral_state.focus >= action.focus_requirement
        
        if not energy_available:
            return max(0.0, behavioral_state.energy / action.energy_requirement)
        
        if not focus_available:
            return max(0.0, behavioral_state.focus / action.focus_requirement)
        
        return 1.0  # No constraints
    
    def _calculate_social_dynamics(self, behavioral_state: BehavioralState, 
                                 action: Action, context: Dict[str, Any]) -> float:
        """Factor 8: Social dynamics (+1.5). DISABLED FOR MAZE - no social interactions."""
        return 0.0  # Simplified: maze has no social dynamics
    
    def _calculate_learning_opportunities(self, action: Action, context: Dict[str, Any]) -> float:
        """Factor 9: Learning opportunities (+1.0). SIMPLIFIED FOR MAZE - only exploration matters."""
        novelty = context.get('novelty_factor', 0.0)
        information_gain = context.get('information_gain', 0.0)
        
        # Only basic exploration - no creative solutions or investigations
        if action.action_type == ActionType.EXPLORE:
            return (novelty + information_gain) * 1.0
        
        return 0.0
    
    def _calculate_risk_assessment(self, behavioral_state: BehavioralState, action: Action) -> float:
        """Factor 10: Risk assessment (-2.0 to +2.0)."""
        risk_tolerance = behavioral_state.risk_tolerance
        action_risk = action.risk_level
        
        # Risk-tolerant agents get bonus for risky actions, penalty for safe ones
        # Risk-averse agents get penalty for risky actions, bonus for safe ones
        risk_differential = action_risk - 0.5  # Center around 0.5 risk
        risk_preference = risk_tolerance - 0.5  # Center around 0.5 tolerance
        
        # Aligned preferences get bonus, misaligned get penalty
        risk_alignment = risk_preference * risk_differential
        
        return risk_alignment * 4.0  # Scale to -2.0 to +2.0 range
    
    def _calculate_consciousness_modifier(self, consciousness: ConsciousnessState,
                                        behavioral_state: BehavioralState, 
                                        action: Action) -> float:
        """Factor 11: Consciousness modifier (behavioral state influence)."""
        modifier = 1.0
        
        # Social actions benefit from social drive
        if action.interaction_type == InteractionType.SOCIAL:
            modifier *= (0.5 + behavioral_state.social_drive * 1.5)
        
        # Combat/Exploration benefit from risk tolerance
        elif action.interaction_type in [InteractionType.COMBAT, InteractionType.EXPLORATION]:
            modifier *= (0.5 + behavioral_state.risk_tolerance * 1.5)
        
        # Economic/Learning benefit from ambition
        elif action.interaction_type in [InteractionType.ECONOMIC, InteractionType.LEARNING]:
            modifier *= (0.5 + behavioral_state.ambition * 1.5)
        
        # Creative actions benefit from creativity
        elif action.interaction_type == InteractionType.CREATIVE:
            modifier *= (0.5 + behavioral_state.creativity * 1.5)
        
        return max(0.1, modifier)
    
    def _calculate_temporal_factors(self, action: Action, context: Dict[str, Any]) -> float:
        """Factor 12: Temporal factors (+0.5)."""
        time_pressure = context.get('time_pressure', 0.0)
        action_duration = context.get('action_duration', 1.0)
        
        # Quick actions favored under time pressure
        if time_pressure > 0.5 and action_duration < 0.5:
            return 0.5
        
        # Slow, careful actions favored when no time pressure
        elif time_pressure < 0.3 and action_duration > 0.7:
            return 0.3
        
        return 0.0
    
    def _calculate_quantum_resonance(self, consciousness: ConsciousnessState, action: Action) -> float:
        """Factor 13: Quantum resonance (coherence-based)."""
        # Map consciousness frequency to action energy
        character_energy = consciousness.frequency
        action_energy = self._map_action_to_energy(action)
        
        # Calculate resonance using quantum-inspired formula
        energy_diff = character_energy - action_energy
        resonance = math.exp(-math.pow(energy_diff - self.gamma_freq, 2) / (2 * self.gamma_freq))
        
        # Coherence amplifies resonance effect
        coherence_amplifier = 0.5 + consciousness.coherence * 0.5
        
        return resonance * coherence_amplifier
    
    def _map_action_to_energy(self, action: Action) -> float:
        """Map action to energy level for resonance calculation."""
        energy_map = {
            ActionType.REST: 3.0,
            ActionType.AVOID: 4.0,
            ActionType.BACKTRACK: 5.0,
            ActionType.TURN_LEFT: 6.0,
            ActionType.TURN_RIGHT: 6.0,
            ActionType.MOVE_FORWARD: 8.0,
            ActionType.INVESTIGATE: 9.0,
            ActionType.EXPLORE: 10.0,
            ActionType.SOCIALIZE: 11.0,
            ActionType.CREATIVE_SOLUTION: 12.0
        }
        return energy_map.get(action.action_type, 7.0)
    
    def select_action(self, actions: List[Action], weights: List[float]) -> Tuple[Action, float]:
        """
        Select action using temperature-based softmax for exploration.
        
        Args:
            actions: List of available actions
            weights: Corresponding weights for each action
            
        Returns:
            Tuple of (selected_action, selection_probability)
        """
        if not actions or not weights:
            raise ValueError("Actions and weights lists cannot be empty")
        
        # Apply temperature to weights for softmax
        temp_weights = [w / self.temperature for w in weights]
        
        # Softmax calculation with numerical stability
        max_weight = max(temp_weights)
        exp_weights = [math.exp(w - max_weight) for w in temp_weights]
        sum_exp = sum(exp_weights)
        probabilities = [w / sum_exp for w in exp_weights]
        
        # Select action based on probability distribution
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return actions[i], prob
        
        # Fallback (shouldn't happen with proper probabilities)
        return actions[-1], probabilities[-1]
    
    def boost_temperature_after_failure(self, boost_duration: int = 7):
        """
        Temporarily boost temperature after failure for increased exploration.
        
        Args:
            boost_duration: Number of steps to maintain boosted temperature
        """
        self.temperature = min(2.0, self.base_temperature + 1.0)
        self.failure_steps_remaining = boost_duration
    
    def update_temperature(self):
        """Update temperature (called each step)."""
        if self.failure_steps_remaining > 0:
            self.failure_steps_remaining -= 1
            if self.failure_steps_remaining == 0:
                self.temperature = self.base_temperature
    
    def anneal_temperature(self, step: int, total_steps: int):
        """
        Annealing schedule for temperature over time.
        
        Args:
            step: Current step number
            total_steps: Total steps in episode
        """
        # Linear annealing from initial to 0.5x initial
        progress = step / total_steps
        self.base_temperature = self.base_temperature * (1.0 - 0.5 * progress)
        
        # Update current temperature if not in failure boost mode
        if self.failure_steps_remaining == 0:
            self.temperature = self.base_temperature
    
    def get_decision_breakdown(self, weights: List[float], factors: Dict[str, float]) -> Dict[str, Any]:
        """
        Get detailed breakdown of decision factors for analysis/debugging.
        
        Args:
            weights: Final action weights
            factors: Dictionary of factor contributions
            
        Returns:
            Decision analysis breakdown
        """
        return {
            'total_actions': len(weights),
            'weight_range': {'min': min(weights), 'max': max(weights), 'avg': sum(weights)/len(weights)},
            'temperature': self.temperature,
            'failure_boost_remaining': self.failure_steps_remaining,
            'factor_contributions': factors,
            'top_action_weight': max(weights),
            'weight_distribution': {
                'std': self._calculate_std(weights),
                'entropy': self._calculate_entropy(weights)
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _calculate_entropy(self, weights: List[float]) -> float:
        """Calculate entropy of weight distribution."""
        # Convert weights to probabilities
        total = sum(weights)
        if total == 0:
            return 0.0
        
        probs = [w / total for w in weights]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return entropy