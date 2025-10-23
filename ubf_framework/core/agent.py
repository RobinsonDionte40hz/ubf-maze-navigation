"""
Universal Behavioral Framework - Agent

Main agent class that integrates consciousness, behavioral state, memory, 
and decision-making systems.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple

from .consciousness_state import ConsciousnessState, BehavioralState
from .memory_system import MemoryManager, InteractionType
from .decision_system import DecisionSystem, Action, ActionType
from .consciousness_update import ConsciousnessUpdateService, EventData, NoiseScheduler
from .collective_memory import CollectiveMemoryPool


class Agent:
    """
    Universal intelligent agent using the UBF framework.
    Combines consciousness coordinates, behavioral state, memory, and decision-making.
    """
    
    def __init__(self, 
                 agent_id: str,
                 initial_frequency: float = 7.5,
                 initial_coherence: float = 0.7,
                 temperature: float = 1.0,
                 collective_memory: Optional['CollectiveMemoryPool'] = None):
        """
        Initialize agent with consciousness coordinates.
        
        Args:
            agent_id: Unique identifier
            initial_frequency: Starting frequency (3-15 Hz)
            initial_coherence: Starting coherence (0.2-1.0)
            temperature: Decision temperature for exploration
            collective_memory: Optional shared memory pool for group learning
        """
        self.agent_id = agent_id
        
        # Core systems
        self.consciousness = ConsciousnessState(initial_frequency, initial_coherence)
        self.behavioral_state = BehavioralState.from_consciousness(self.consciousness)
        self.memory_manager = MemoryManager()
        self.decision_system = DecisionSystem(temperature=temperature)
        self.update_service = ConsciousnessUpdateService()
        
        # Collective learning
        self.collective_memory = collective_memory
        self.broadcast_threshold = 0.12  # Minimum significance to share real-time
        self.last_broadcast_position = None  # Track to avoid spamming same location
        
        # Agent state
        self.position = (0, 0)  # Current position in environment
        self.orientation = 0    # Direction facing (0=North, 1=East, 2=South, 3=West)
        self.step_count = 0
        self.last_action = None
        self.last_action_outcome = None
        
        # Performance tracking
        self.goal_achieved = False
        self.total_reward = 0.0
        self.action_history = []
        self.respawn_count = 0  # How many times this agent has restarted
        
        # Successful path tracking - remember the furthest progress
        self.best_distance_to_goal = 9999  # Track best progress
        self.current_path = []  # Current run's path: [(pos, dir, action), ...]
        self.best_path_positions = set()  # Positions from best run
        
        # SIMPLE DIRECT TRACKING - override complex memory system
        self.wall_positions = set()  # (pos, direction) tuples where walls exist
        self.successful_moves = {}  # pos -> set of directions that worked from there
        
    def get_available_actions(self, environment_context: Dict[str, Any]) -> List[Action]:
        """
        Get list of available actions based on current environment.
        
        Args:
            environment_context: Current environment state
            
        Returns:
            List of available Action objects
        """
        actions = []
        
        # Basic movement actions (always available)
        moving_toward_goal = environment_context.get('moving_toward_goal', False)
        can_see_exit = environment_context.get('can_see_exit', False)
        adjacent_clear = environment_context.get('adjacent_clear', {})
        
        # Get current orientation to determine what left/right mean
        current_direction = environment_context.get('current_direction', 0)  # 0=N, 1=E, 2=S, 3=W
        
        # Calculate which directions left and right would face
        left_direction = (current_direction - 1) % 4
        right_direction = (current_direction + 1) % 4
        
        # Check if turning would lead to clear paths
        left_is_clear = adjacent_clear.get(left_direction, False)
        right_is_clear = adjacent_clear.get(right_direction, False)
        forward_is_clear = adjacent_clear.get(current_direction, False)
        
        # SIMPLIFIED FOR MAZE: Only 3 actions needed
        actions = [
            Action(
                action_type=ActionType.MOVE_FORWARD,
                interaction_type=InteractionType.EXPLORATION,
                base_weight=1.0,
                energy_requirement=0.1,
                focus_requirement=0.1,
                risk_level=0.2,
                goal_alignment=0.6,
                description="Move forward"
            ),
            Action(
                action_type=ActionType.TURN_LEFT,
                interaction_type=InteractionType.EXPLORATION,
                base_weight=1.0,
                energy_requirement=0.05,
                focus_requirement=0.05,
                risk_level=0.1,
                goal_alignment=0.3,
                description="Turn left"
            ),
            Action(
                action_type=ActionType.TURN_RIGHT,
                interaction_type=InteractionType.EXPLORATION,
                base_weight=1.0,
                energy_requirement=0.05,
                focus_requirement=0.05,
                risk_level=0.1,
                goal_alignment=0.3,
                description="Turn right"
            )
        ]
        
        return actions
    
    def select_action(self, environment_context: Dict[str, Any]) -> Tuple[Action, Dict[str, Any]]:
        """
        Select action using the 13-factor decision system.
        
        Args:
            environment_context: Current environment state
            
        Returns:
            Tuple of (selected_action, decision_breakdown)
        """
        # Get available actions
        actions = self.get_available_actions(environment_context)
        
        if not actions:
            raise ValueError("No available actions")
        
        # STUCK DETECTION: If stuck near start, force MOVE_FORWARD heavily
        stuck_count = environment_context.get('stuck_count', 0)
        near_start = (abs(self.position[0] - 1) <= 2 and abs(self.position[1] - 1) <= 2)
        
        if stuck_count > 3 and near_start:  # Changed from 5 to 3 for earlier intervention
            # Force forward movement to break out of spawn spiral
            for i, action in enumerate(actions):
                if action.action_type == ActionType.MOVE_FORWARD:
                    # Just pick forward movement directly
                    decision_breakdown = {
                        'selected_action': action.action_type.value,
                        'selection_probability': 1.0,
                        'reason': 'forced_exploration_stuck_at_start'
                    }
                    return action, decision_breakdown
        
        # Calculate weights for each action
        weights = []
        factor_contributions = {}
        
        for i, action in enumerate(actions):
            weight = self.decision_system.calculate_interaction_weight(
                consciousness=self.consciousness,
                behavioral_state=self.behavioral_state,
                memory_manager=self.memory_manager,
                action=action,
                environmental_context=environment_context,
                collective_memory=self.collective_memory  # Pass collective memory
            )
            
            # DIRECT WALL AVOIDANCE - if we know there's a wall, heavily penalize!
            if action.action_type == ActionType.MOVE_FORWARD:
                dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                dx, dy = dir_vectors[self.orientation]
                next_pos = (self.position[0] + dx, self.position[1] + dy)
                
                # Check if we know there's a wall there
                if (next_pos, self.orientation) in self.wall_positions:
                    weight *= 0.1  # MASSIVE penalty - we KNOW this is a wall!
                    
                # Check if this is on our best path
                elif next_pos in self.best_path_positions:
                    weight *= 2.5  # Strong boost for known good path
            
            # SMART TURNING - if we know which directions work from here, boost those turns
            if action.action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
                if self.position in self.successful_moves:
                    good_directions = self.successful_moves[self.position]
                    
                    # Calculate which direction we'd face after this turn
                    if action.action_type == ActionType.TURN_LEFT:
                        new_direction = (self.orientation - 1) % 4
                    else:
                        new_direction = (self.orientation + 1) % 4
                    
                    if new_direction in good_directions:
                        weight *= 2.0  # Boost turns toward any known good direction!
            
            weights.append(weight)
        
        # SYNC TEMPERATURE FROM NOISE SCHEDULER (for coherence recovery)
        recovery_temp = self.update_service.noise_scheduler.get_temperature()
        if recovery_temp > 1.0:
            self.decision_system.temperature = recovery_temp
        
        # Select action using temperature-based softmax
        selected_action, selection_probability = self.decision_system.select_action(actions, weights)
        
        # Update temperature (annealing, failure boost decay)
        self.decision_system.update_temperature()
        
        # Create decision breakdown for analysis
        decision_breakdown = self.decision_system.get_decision_breakdown(weights, factor_contributions)
        decision_breakdown['selected_action'] = selected_action.action_type.value
        decision_breakdown['selection_probability'] = selection_probability
        
        return selected_action, decision_breakdown
    
    def execute_action(self, action: Action, environment_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action and process the outcome.
        
        Args:
            action: Action to execute
            environment_context: Current environment state
            
        Returns:
            Action result with outcome information
        """
        self.step_count += 1
        self.last_action = action
        
        # Simulate action execution (environment-specific)
        # This would be overridden by specific environment implementations
        result = {
            'action': action.action_type.value,
            'success': True,
            'outcome': 'success',
            'reward': 0.1,
            'position_change': (0, 0),
            'new_information': False,
            'context': environment_context
        }
        
        # Update position if movement action
        if action.action_type == ActionType.MOVE_FORWARD:
            # This would be handled by environment
            pass
        elif action.action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
            # Update orientation
            if action.action_type == ActionType.TURN_LEFT:
                self.orientation = (self.orientation - 1) % 4
            else:
                self.orientation = (self.orientation + 1) % 4
        
        # Store action in history
        self.action_history.append({
            'step': self.step_count,
            'action': action.action_type.value,
            'outcome': result['outcome'],
            'reward': result['reward'],
            'consciousness': self.consciousness.to_dict(),
            'behavioral_state': self.behavioral_state.to_dict()
        })
        
        self.last_action_outcome = result
        return result
    
    def process_outcome(self, action_result: Dict[str, Any]):
        """
        Process action outcome and update consciousness/memory if significant.
        
        Args:
            action_result: Result from action execution
        """
        if self.last_action is None:
            return  # No action to process
        
        # Calculate target position for forward movement (for collision memories)
        memory_location = f"pos_{self.position}"  # Default: current position
        if (self.last_action.action_type == ActionType.MOVE_FORWARD and 
            action_result['outcome'] == 'collision'):
            # Store collision memory at the TARGET position (the wall location)
            # This way future attempts to move there will be avoided
            dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
            dx, dy = dir_vectors[self.orientation]
            target_x = self.position[0] + dx
            target_y = self.position[1] + dy
            memory_location = f"pos_{(target_x, target_y)}"
            
        # DIRECT TRACKING - record walls and successful moves immediately!
        if self.last_action.action_type == ActionType.MOVE_FORWARD:
            if action_result['outcome'] == 'collision':
                # Remember this wall!
                dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                dx, dy = dir_vectors[self.orientation]
                wall_pos = (self.position[0] + dx, self.position[1] + dy)
                self.wall_positions.add((wall_pos, self.orientation))
                
            elif action_result['outcome'] == 'success':
                # Remember this successful move!
                old_pos = action_result.get('context', {}).get('old_position', self.position)
                if old_pos not in self.successful_moves:
                    self.successful_moves[old_pos] = set()
                self.successful_moves[old_pos].add(self.orientation)
        
        # Create event data from action result
        event_data = self.update_service.create_event_from_action_outcome(
            action_type=self.last_action.action_type.value,
            outcome=action_result['outcome'],
            interaction_type=self.last_action.interaction_type,
            context={
                'novelty_factor': action_result.get('new_information', False) * 0.8,  # Boosted from 0.5
                'goal_relevance': self.last_action.goal_alignment,
                'importance': abs(action_result.get('reward', 0.0)) * 1.5,  # Amplify importance
                'surprise_factor': action_result.get('surprise', 0.0),
                'location': memory_location,  # Use target location for collisions!
                'participants': [],
                'tags': [action_result['outcome'], f"dir_{self.orientation}", 
                        f"action_{self.last_action.action_type.value}"],
                'description': f"Action {self.last_action.action_type.value} at {self.position} facing {self.orientation} resulted in {action_result['outcome']}"
            }
        )
        
        # Process event and update consciousness if significant
        was_updated, new_behavioral_state = self.update_service.process_event(
            consciousness=self.consciousness,
            memory_manager=self.memory_manager,
            event_data=event_data
        )
        
        # Update behavioral state if consciousness changed
        if was_updated and new_behavioral_state:
            self.behavioral_state = new_behavioral_state
            
            # Boost temperature after failures for increased exploration
            if action_result['outcome'] == 'failure':
                self.decision_system.boost_temperature_after_failure()
            
            # REAL-TIME BROADCASTING: Share significant experiences immediately
            # This allows agents to learn from each other's successes AND failures as they happen
            if self.collective_memory is not None and was_updated:
                self.broadcast_recent_experience(event_data, action_result)
        
        # Update total reward
        self.total_reward += action_result.get('reward', 0.0)
        
        # If goal achieved, broadcast all key memories for complete path knowledge
        if (action_result.get('outcome') == 'goal_achieved' and 
            self.collective_memory is not None):
            self.broadcast_success_to_collective()
    
    def broadcast_recent_experience(self, event_data, action_result):
        """
        Share significant recent experiences in real-time (both successes AND failures).
        This enables continuous learning across the group as events happen.
        
        Args:
            event_data: The event that was just processed
            action_result: The outcome of the action
        """
        if self.collective_memory is None:
            return
        
        # Get the most recent memory that was just created (if any)
        # Look for memories at the location where the event occurred
        event_location = event_data.location if event_data.location else f"pos_{self.position}"
        recent_memories = [m for m in self.memory_manager.memories 
                          if m.location == event_location]
        
        if not recent_memories:
            return
        
        # Get the newest memory at this location
        latest_memory = max(recent_memories, key=lambda m: m.timestamp)
        
        # Only broadcast if it's significant enough
        if latest_memory.weighted_significance < self.broadcast_threshold:
            return
        
        # Don't spam broadcasts from the same position (unless outcome changed)
        if self.last_broadcast_position == self.position:
            # Only re-broadcast if we learned something NEW at this position
            if action_result.get('outcome') not in ['collision', 'goal_achieved']:
                return
        
        # Determine what kind of experience this is
        outcome = action_result.get('outcome', 'unknown')
        
        # Share immediately so other agents can learn from it
        # - Collisions: "Don't go this way at this position!"
        # - Successes: "This direction worked here!"
        # - New discoveries: "Found something interesting here!"
        contributed = self.collective_memory.broadcast_success_memories(
            agent_id=self.agent_id,
            memories=[latest_memory],
            context={
                'broadcast_type': 'real_time',
                'outcome': outcome,
                'position': self.position,
                'direction': self.orientation,
                'step_count': self.step_count
            }
        )
        
        # Track that we broadcast from this position
        if contributed > 0:
            self.last_broadcast_position = self.position
    
    def broadcast_success_to_collective(self):
        """
        Share successful memories with the collective pool when reaching goal.
        Broadcasts the most significant memories, especially those near the end.
        """
        if self.collective_memory is None:
            return
        
        # Get top memories (most significant ones)
        all_memories = self.memory_manager.memories
        
        # Sort by significance and recency (recent memories likely more relevant to success)
        sorted_memories = sorted(
            all_memories,
            key=lambda m: m.weighted_significance * (1.0 + m.timestamp * 0.1),
            reverse=True
        )
        
        # Share top 20 memories (or all if fewer)
        memories_to_share = sorted_memories[:20]
        
        context = {
            'final_position': self.position,
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'respawn_count': self.respawn_count
        }
        
        contributed = self.collective_memory.broadcast_success_memories(
            self.agent_id,
            memories_to_share,
            context
        )
        
        return contributed
    
    def respawn(self, keep_personal_memories: bool = True, spawn_position: tuple = None):
        """
        Respawn agent at starting position while keeping learned knowledge.
        This is different from reset_for_new_run - it's meant for continuous learning
        where the agent dies/fails but wants to try again with what they learned.
        
        Args:
            keep_personal_memories: Whether to keep personal memory (almost always True)
            spawn_position: Position to respawn at (defaults to (1,1) if not provided)
        """
        # Reset position and state - spawn_position should be provided by environment
        self.position = spawn_position if spawn_position else (1, 1)
        
        # SMART ORIENTATION: Use memories to pick best starting direction!
        # Check which direction from spawn has the best memories
        best_orientation = 0
        best_score = -999.0
        
        for direction in range(4):  # Check all 4 directions
            dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
            dx, dy = dir_vectors[direction]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            check_location = f"pos_{next_pos}"
            
            # Query memories at that location
            from .memory_system import InteractionType
            memories = self.memory_manager.retrieve_relevant_memories(InteractionType.EXPLORATION, 5)
            memories_at_location = [m for m in memories if m.location == check_location]
            
            # Calculate average emotional impact
            if memories_at_location:
                avg_impact = sum(m.emotional_impact for m in memories_at_location) / len(memories_at_location)
                if avg_impact > best_score:
                    best_score = avg_impact
                    best_orientation = direction
        
        # Use the direction with best memories, or East (1) if no memories
        self.orientation = best_orientation if best_score > -999.0 else 1
        
        self.step_count = 0
        self.goal_achieved = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_action_outcome = None
        self.action_history = []
        
        # Reset current path but KEEP best path positions (learned route)
        self.current_path = []
        # best_distance_to_goal and best_path_positions are NOT reset!
        
        # Reset decision temperature
        self.decision_system.temperature = self.decision_system.base_temperature
        self.decision_system.failure_steps_remaining = 0
        
        # Keep memories (both personal and access to collective)
        # Personal memories are NOT cleared
        if not keep_personal_memories:
            self.memory_manager.clear_memories()
        
        # Increment respawn counter
        self.respawn_count += 1
        
        # MINIMAL coherence penalty - we want them to stay focused and learn!
        # Reduced from -0.05 to -0.02 so they don't get too scattered
        self.consciousness.update_coordinates(0.0, -0.02, noise_std=0.01)
        self.behavioral_state = BehavioralState.from_consciousness(self.consciousness)
    
    def reset_for_new_run(self, keep_memories: bool = True, keep_consciousness: bool = True):
        """
        Reset agent for a new run while optionally preserving learning.
        
        Args:
            keep_memories: Whether to preserve memory from previous runs
            keep_consciousness: Whether to preserve consciousness coordinates
        """
        # Reset position and state
        self.position = (0, 0)
        self.orientation = 0
        self.step_count = 0
        self.goal_achieved = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_action_outcome = None
        self.action_history = []
        
        # Reset decision temperature
        self.decision_system.temperature = self.decision_system.base_temperature
        self.decision_system.failure_steps_remaining = 0
        
        # Optionally reset memories
        if not keep_memories:
            self.memory_manager.clear_memories()
        
        # Optionally reset consciousness (for naive agents)
        if not keep_consciousness:
            self.consciousness = ConsciousnessState(7.5, 0.7)
            self.behavioral_state = BehavioralState.from_consciousness(self.consciousness)
        else:
            # Just regenerate behavioral state from current consciousness
            self.behavioral_state = BehavioralState.from_consciousness(self.consciousness)
    
    def share_experience_with(self, other_agent: 'Agent') -> Dict[str, Any]:
        """
        Share experience with another agent (for group scenarios).
        
        Args:
            other_agent: Agent to share experience with
            
        Returns:
            Information about what was shared
        """
        # Share averaged consciousness coordinates
        avg_freq = (self.consciousness.frequency + other_agent.consciousness.frequency) / 2
        avg_coh = (self.consciousness.coherence + other_agent.consciousness.coherence) / 2
        
        # Each agent moves 50% toward the average
        freq_delta = (avg_freq - self.consciousness.frequency) * 0.5
        coh_delta = (avg_coh - self.consciousness.coherence) * 0.5
        
        self.consciousness.update_coordinates(freq_delta, coh_delta, noise_std=0.05)
        other_agent.consciousness.update_coordinates(-freq_delta, -coh_delta, noise_std=0.05)
        
        # Regenerate behavioral states
        self.behavioral_state = BehavioralState.from_consciousness(self.consciousness)
        other_agent.behavioral_state = BehavioralState.from_consciousness(other_agent.consciousness)
        
        return {
            'shared_frequency_delta': freq_delta,
            'shared_coherence_delta': coh_delta,
            'agents': [self.agent_id, other_agent.agent_id]
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary for logging/analysis."""
        return {
            'agent_id': self.agent_id,
            'step_count': self.step_count,
            'position': self.position,
            'orientation': self.orientation,
            'goal_achieved': self.goal_achieved,
            'total_reward': self.total_reward,
            'consciousness': self.consciousness.to_dict(),
            'behavioral_state': self.behavioral_state.to_dict(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'decision_temperature': self.decision_system.temperature,
            'last_action': self.last_action.action_type.value if self.last_action else None,
            'last_outcome': self.last_action_outcome.get('outcome') if self.last_action_outcome else None
        }
    
    def get_detailed_history(self) -> Dict[str, Any]:
        """Get detailed action history for analysis."""
        return {
            'agent_id': self.agent_id,
            'final_status': self.get_status_summary(),
            'action_history': self.action_history,
            'memory_details': self.memory_manager.to_dict(),
            'consciousness_trajectory': self._extract_consciousness_trajectory()
        }
    
    def _extract_consciousness_trajectory(self) -> List[Dict[str, Any]]:
        """Extract consciousness coordinate changes over time."""
        trajectory = []
        for entry in self.action_history:
            trajectory.append({
                'step': entry['step'],
                'frequency': entry['consciousness']['frequency'],
                'coherence': entry['consciousness']['coherence'],
                'energy_level': entry['consciousness']['energy_level'],
                'focus_level': entry['consciousness']['focus_level']
            })
        return trajectory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'consciousness': self.consciousness.to_dict(),
            'behavioral_state': self.behavioral_state.to_dict(),
            'memory_system': self.memory_manager.to_dict(),
            'position': self.position,
            'orientation': self.orientation,
            'step_count': self.step_count,
            'goal_achieved': self.goal_achieved,
            'total_reward': self.total_reward
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary."""
        agent = cls(
            agent_id=data['agent_id'],
            initial_frequency=data['consciousness']['frequency'],
            initial_coherence=data['consciousness']['coherence']
        )
        
        # Restore state
        agent.consciousness = ConsciousnessState.from_dict(data['consciousness'])
        agent.behavioral_state = BehavioralState.from_consciousness(agent.consciousness)
        agent.memory_manager.from_dict(data['memory_system'])
        agent.position = tuple(data['position'])
        agent.orientation = data['orientation']
        agent.step_count = data['step_count']
        agent.goal_achieved = data['goal_achieved']
        agent.total_reward = data['total_reward']
        
        return agent
    
    def __str__(self) -> str:
        return f"Agent({self.agent_id}: {self.consciousness}, pos={self.position}, steps={self.step_count})"