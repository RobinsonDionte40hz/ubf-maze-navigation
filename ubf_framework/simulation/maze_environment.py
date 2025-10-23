"""
Universal Behavioral Framework - Maze Simulation

2D maze environment with triangular line-of-sight, DFS carving,
and agent navigation for testing the UBF system.
"""

import math
import random
import time
import sys
import os
from typing import List, Tuple, Dict, Any, Optional, Set
from enum import Enum

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from core.decision_system import ActionType
from core.memory_system import InteractionType


class CellType(Enum):
    """Types of cells in the maze."""
    WALL = "#"
    EMPTY = " "
    START = "S"
    EXIT = "E"
    VISITED = "."
    AGENT = "A"


class Direction(Enum):
    """Cardinal directions."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class MazeEnvironment:
    """
    2D maze environment with intelligent agent navigation.
    Features DFS-carved mazes, triangular line-of-sight, and UBF integration.
    """
    
    def __init__(self, width: int = 10, height: int = 10, wall_density: float = 0.3, max_steps: int = None):
        """
        Initialize maze environment.
        
        Args:
            width: Maze width (must be odd for DFS carving)
            height: Maze height (must be odd for DFS carving)
            wall_density: Density of random walls (0.0-1.0)
            max_steps: Maximum steps per episode (None for unlimited)
        """
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.wall_density = wall_density
        
        # Maze grid and agent tracking
        self.grid = [[CellType.WALL for _ in range(self.width)] for _ in range(self.height)]
        self.start_pos = (1, 1)  # Top-left accessible area
        self.exit_pos = (self.width - 2, self.height - 2)  # Bottom-right
        self.agent_positions = {}  # agent_id -> position
        self.visited_cells = set()  # Cells visited by any agent
        
        # Environment state
        self.step_count = 0
        self.max_steps = max_steps  # None means no limit!
        self.agents_finished = set()
        
        # Generate maze
        self._generate_maze()
        
        # Line of sight parameters
        self.sight_distance = 5
        self.sight_angle = 60  # degrees (30 degrees each side)
    
    def _generate_maze(self):
        """Generate maze using DFS carving algorithm."""
        # Start with all walls
        self.grid = [[CellType.WALL for _ in range(self.width)] for _ in range(self.height)]
        
        # DFS carving starting from start position
        self._carve_maze(self.start_pos[0], self.start_pos[1])
        
        # Ensure exit is accessible
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = CellType.EXIT
        
        # Add random walls based on density
        self._add_random_walls()
        
        # Verify path exists from start to exit
        if not self._verify_solvable():
            # Regenerate if not solvable
            self._generate_maze()
    
    def _carve_maze(self, x: int, y: int):
        """Recursively carve maze using DFS algorithm."""
        # Mark current cell as empty
        self.grid[y][x] = CellType.EMPTY
        
        # Get directions in random order
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]  # N, E, S, W (skip 1 cell)
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if next position is valid and unvisited
            if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and 
                self.grid[ny][nx] == CellType.WALL):
                
                # Carve path to next cell
                self.grid[y + dy // 2][x + dx // 2] = CellType.EMPTY
                self._carve_maze(nx, ny)
    
    def _add_random_walls(self):
        """Add random walls to increase difficulty."""
        empty_cells = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y][x] == CellType.EMPTY:
                    empty_cells.append((x, y))
        
        # Add walls to a percentage of empty cells
        num_walls = int(len(empty_cells) * self.wall_density * 0.3)  # Limited to preserve solvability
        wall_positions = random.sample(empty_cells, min(num_walls, len(empty_cells)))
        
        for x, y in wall_positions:
            # Don't block start or exit
            if (x, y) not in [self.start_pos, self.exit_pos]:
                self.grid[y][x] = CellType.WALL
    
    def _verify_solvable(self) -> bool:
        """Verify that maze has a path from start to exit using BFS."""
        from collections import deque
        
        queue = deque([self.start_pos])
        visited = {self.start_pos}
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == self.exit_pos:
                return True
            
            # Check adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited and 
                    self.grid[ny][nx] in [CellType.EMPTY, CellType.EXIT]):
                    
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False
    
    def add_agent(self, agent: Agent) -> bool:
        """
        Add agent to maze at start position.
        
        Args:
            agent: Agent to add
            
        Returns:
            True if successfully added
        """
        if agent.agent_id in self.agent_positions:
            return False
        
        agent.position = self.start_pos
        agent.orientation = Direction.EAST.value  # Face right initially
        self.agent_positions[agent.agent_id] = self.start_pos
        
        return True
    
    def remove_agent(self, agent_id: str):
        """Remove agent from maze."""
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
    
    def respawn_agent(self, agent: Agent):
        """
        Respawn an agent at the start position with memories intact.
        
        Args:
            agent: Agent to respawn
        """
        # Call agent's respawn with correct start position
        agent.respawn(keep_personal_memories=True, spawn_position=self.start_pos)
        
        # Update environment tracking
        self.agent_positions[agent.agent_id] = self.start_pos
        
        # Remove from finished set if they were there
        self.agents_finished.discard(agent.agent_id)
    
    def get_line_of_sight(self, agent: Agent) -> Dict[str, Any]:
        """
        Calculate triangular line of sight for agent.
        
        Args:
            agent: Agent to calculate sight for
            
        Returns:
            Dictionary with visible information
        """
        x, y = agent.position
        orientation = agent.orientation
        
        visible_cells = set()
        obstacles_ahead = False
        walls_nearby = 0
        unexplored_nearby = 0
        
        # Direction vectors
        dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = dir_vectors[orientation]
        
        # Check immediately adjacent cells (for tactical awareness)
        adjacent_clear = {}
        for direction, (dx_check, dy_check) in enumerate(dir_vectors):
            adj_x, adj_y = x + dx_check, y + dy_check
            if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                adjacent_clear[direction] = (self.grid[adj_y][adj_x] != CellType.WALL)
            else:
                adjacent_clear[direction] = False
        
        # Calculate sight cone
        for distance in range(1, self.sight_distance + 1):
            # Calculate width of sight cone at this distance
            cone_width = int(distance * math.tan(math.radians(self.sight_angle / 2)))
            
            for offset in range(-cone_width, cone_width + 1):
                # Calculate cell position in sight cone
                if orientation in [0, 2]:  # North/South
                    check_x = x + offset
                    check_y = y + dy * distance
                else:  # East/West
                    check_x = x + dx * distance
                    check_y = y + offset
                
                # Check if position is valid
                if (0 <= check_x < self.width and 0 <= check_y < self.height):
                    visible_cells.add((check_x, check_y))
                    
                    # Check for obstacles and features
                    cell_type = self.grid[check_y][check_x]
                    
                    if cell_type == CellType.WALL:
                        walls_nearby += 1
                        if distance == 1 and offset == 0:  # Directly ahead
                            obstacles_ahead = True
                    elif (check_x, check_y) not in self.visited_cells:
                        unexplored_nearby += 1
        
        return {
            'visible_cells': visible_cells,
            'obstacles_ahead': obstacles_ahead,
            'walls_nearby': walls_nearby,
            'unexplored_areas': unexplored_nearby > 0,
            'unexplored_count': unexplored_nearby,
            'visibility': 1.0,  # Perfect visibility in this simple maze
            'sight_distance': self.sight_distance,
            'can_see_exit': (self.exit_pos in visible_cells),  # Can agent see the goal?
            'adjacent_clear': adjacent_clear  # Which adjacent cells are open (for smart turning)
        }
    
    def execute_agent_action(self, agent: Agent, action_type: ActionType) -> Dict[str, Any]:
        """
        Execute agent action in maze environment.
        
        Args:
            agent: Agent performing action
            action_type: Type of action to execute
            
        Returns:
            Action result with outcome and rewards
        """
        current_pos = agent.position
        x, y = current_pos
        
        result = {
            'action': action_type.value,
            'success': False,
            'outcome': 'failure',
            'reward': -0.01,  # Small step penalty
            'position_change': (0, 0),
            'new_information': False,
            'surprise': 0.0,
            'context': {}
        }
        
        if action_type == ActionType.MOVE_FORWARD:
            # Calculate new position based on orientation
            dir_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
            dx, dy = dir_vectors[agent.orientation]
            new_x, new_y = x + dx, y + dy
            
            # Check if move is valid
            if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                self.grid[new_y][new_x] != CellType.WALL):
                
                # Valid move
                old_pos = agent.position
                agent.position = (new_x, new_y)
                self.agent_positions[agent.agent_id] = agent.position
                
                # Mark cell as visited
                was_unvisited = (new_x, new_y) not in self.visited_cells
                self.visited_cells.add((new_x, new_y))
                
                # Calculate distance improvement
                old_dist = abs(old_pos[0] - self.exit_pos[0]) + abs(old_pos[1] - self.exit_pos[1])
                new_dist = abs(new_x - self.exit_pos[0]) + abs(new_y - self.exit_pos[1])
                getting_closer = new_dist < old_dist
                
                # REWARD STRUCTURE: heavily reward progress
                base_reward = 0.5 if was_unvisited else 0.1  # Strong boost for exploration
                if getting_closer:
                    base_reward += 0.3  # Extra for moving toward goal
                
                result.update({
                    'success': True,
                    'outcome': 'success',
                    'reward': base_reward,
                    'position_change': (dx, dy),
                    'new_information': was_unvisited,
                    'surprise': 0.6 if was_unvisited else 0.1,  # High surprise for new areas
                    'old_position': old_pos  # Track where we moved FROM
                })
                
                # Check if reached exit
                if (new_x, new_y) == self.exit_pos:
                    result.update({
                        'outcome': 'goal_achieved',
                        'reward': 10.0,  # Large reward for completion
                        'goal_completed': True
                    })
                    agent.goal_achieved = True
                    self.agents_finished.add(agent.agent_id)
                
            else:
                # Invalid move (hit wall) - STRONG penalty so they learn!
                result.update({
                    'outcome': 'collision',
                    'reward': -0.5,  # Increased from -0.1 to -0.5 for stronger learning
                    'surprise': 0.8  # Increased from 0.3 to 0.8 for high surprise
                })
        
        elif action_type == ActionType.TURN_LEFT:
            agent.orientation = (agent.orientation - 1) % 4
            result.update({
                'success': True,
                'outcome': 'success',
                'reward': -0.05  # Increased from -0.005 to make turning expensive
            })
        
        elif action_type == ActionType.TURN_RIGHT:
            agent.orientation = (agent.orientation + 1) % 4
            result.update({
                'success': True,
                'outcome': 'success',
                'reward': -0.05  # Increased from -0.005 to make turning expensive
            })
        
        elif action_type == ActionType.INVESTIGATE:
            # Look around current area more carefully
            sight_info = self.get_line_of_sight(agent)
            new_info = sight_info['unexplored_count'] > 0
            
            result.update({
                'success': True,
                'outcome': 'success',
                'reward': 0.02 if new_info else 0.01,
                'new_information': new_info
            })
        
        elif action_type == ActionType.EXPLORE:
            # Enhanced exploration (like investigate but with movement intent)
            sight_info = self.get_line_of_sight(agent)
            result.update({
                'success': True,
                'outcome': 'success',
                'reward': 0.03,
                'new_information': sight_info['unexplored_count'] > 2
            })
        
        elif action_type == ActionType.REST:
            # Rest to recover energy (beneficial for low-energy agents)
            energy_gain = min(0.2, 1.0 - agent.behavioral_state.energy)
            result.update({
                'success': True,
                'outcome': 'success',
                'reward': energy_gain * 0.1  # Small reward for needed rest
            })
        
        elif action_type == ActionType.CREATIVE_SOLUTION:
            # Attempt creative problem solving
            sight_info = self.get_line_of_sight(agent)
            if sight_info['obstacles_ahead']:
                # Creative solution when facing obstacle
                creativity_success = agent.behavioral_state.creativity > 0.6
                if creativity_success:
                    result.update({
                        'success': True,
                        'outcome': 'breakthrough',
                        'reward': 0.2,
                        'new_information': True,
                        'surprise': 0.5
                    })
                else:
                    result.update({
                        'outcome': 'failed_attempt',
                        'reward': -0.05
                    })
            else:
                result.update({
                    'outcome': 'unnecessary',
                    'reward': -0.02
                })
        
        # Add environmental context to result
        result['context'] = self._get_environmental_context(agent)
        
        return result
    
    def get_environmental_context(self, agent: Agent) -> Dict[str, Any]:
        """Get environmental context for agent decision-making (public method)."""
        return self._get_environmental_context(agent)
    
    def _get_environmental_context(self, agent: Agent) -> Dict[str, Any]:
        """Get current environmental context for agent."""
        sight_info = self.get_line_of_sight(agent)
        
        # Calculate terrain difficulty (walls nearby = harder terrain)
        terrain_difficulty = min(1.0, sight_info['walls_nearby'] / 10.0)
        
        # Calculate goal progress (distance to exit)
        distance_to_exit = self._manhattan_distance(agent.position, self.exit_pos)
        max_distance = self.width + self.height
        goal_progress = 1.0 - (distance_to_exit / max_distance)
        
        # Calculate direction to goal (for goal alignment)
        dx = self.exit_pos[0] - agent.position[0]
        dy = self.exit_pos[1] - agent.position[1]
        
        # Determine if moving forward would get closer to goal
        direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        forward_dx, forward_dy = direction_vectors[agent.orientation]
        
        # Check if forward movement aligns with goal direction
        current_distance = distance_to_exit
        next_x = agent.position[0] + forward_dx
        next_y = agent.position[1] + forward_dy
        next_distance = abs(next_x - self.exit_pos[0]) + abs(next_y - self.exit_pos[1])
        moving_toward_goal = next_distance < current_distance
        
        # Stuck detection
        recent_positions = []
        if len(agent.action_history) >= 5:
            recent_positions = [entry.get('position', agent.position) 
                             for entry in agent.action_history[-5:]]
        stuck_count = len(recent_positions) - len(set(recent_positions))
        
        return {
            'terrain_difficulty': terrain_difficulty,
            'visibility': sight_info['visibility'],
            'obstacles_nearby': sight_info['walls_nearby'] / 10.0,
            'obstacles_ahead': sight_info['obstacles_ahead'],
            'unexplored_areas': sight_info['unexplored_areas'],
            'goal_progress': goal_progress,
            'distance_to_goal': distance_to_exit,
            'moving_toward_goal': moving_toward_goal,
            'goal_urgency': min(1.0, self.step_count / self.max_steps) if self.max_steps else 0.5,
            'stuck_count': stuck_count,
            'danger_level': 0.0,  # No danger in simple maze
            'novelty_factor': sight_info['unexplored_count'] / max(1, sight_info['sight_distance'] ** 2),
            'information_gain': 0.5 if sight_info['unexplored_areas'] else 0.1,
            'time_pressure': (max(0.0, (self.step_count - self.max_steps * 0.7) / (self.max_steps * 0.3)) 
                             if self.max_steps else 0.0),
            'current_location': f"pos_{agent.position}",  # For spatial memory
            'current_direction': agent.orientation  # For directional memory
        }
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, agents: List[Agent]) -> Dict[str, Any]:
        """
        Execute one simulation step for all agents.
        
        Args:
            agents: List of agents to step
            
        Returns:
            Step summary information
        """
        self.step_count += 1
        step_results = {}
        
        for agent in agents:
            if agent.agent_id in self.agents_finished:
                continue  # Skip finished agents
            
            # Get environmental context
            env_context = self._get_environmental_context(agent)
            
            # Agent selects action
            selected_action, decision_breakdown = agent.select_action(env_context)
            
            # Execute action in environment
            action_result = self.execute_agent_action(agent, selected_action.action_type)
            
            # Agent processes outcome and updates internal state
            agent.process_outcome(action_result)
            
            step_results[agent.agent_id] = {
                'action': selected_action.action_type.value,
                'outcome': action_result['outcome'],
                'reward': action_result['reward'],
                'position': agent.position,
                'goal_achieved': agent.goal_achieved,
                'decision_breakdown': decision_breakdown
            }
        
        return {
            'step': self.step_count,
            'agent_results': step_results,
            'environment_state': self._get_environment_state(),
            'simulation_complete': self._check_simulation_complete(agents)
        }
    
    def _get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state summary."""
        return {
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'agents_finished': len(self.agents_finished),
            'total_visited_cells': len(self.visited_cells),
            'exploration_percentage': len(self.visited_cells) / ((self.width - 2) * (self.height - 2))
        }
    
    def _check_simulation_complete(self, agents: List[Agent]) -> bool:
        """Check if simulation is complete."""
        # Complete if all agents finished or max steps reached (if limit set)
        all_finished = len(self.agents_finished) == len(agents)
        max_steps_reached = (self.max_steps is not None and 
                            self.step_count >= self.max_steps)
        return all_finished or max_steps_reached
    
    def reset(self):
        """Reset environment for new run."""
        self.step_count = 0
        self.agent_positions = {}
        self.visited_cells = set()
        self.agents_finished = set()
        
        # Regenerate maze
        self._generate_maze()
    
    def render_ascii(self, agents: List[Agent] = None) -> str:
        """
        Render maze as ASCII art.
        
        Args:
            agents: Optional list of agents to show positions
            
        Returns:
            ASCII representation of maze
        """
        display_grid = [[cell.value for cell in row] for row in self.grid]
        
        # Mark start and exit
        display_grid[self.start_pos[1]][self.start_pos[0]] = CellType.START.value
        display_grid[self.exit_pos[1]][self.exit_pos[0]] = CellType.EXIT.value
        
        # Mark visited cells
        for x, y in self.visited_cells:
            if display_grid[y][x] == CellType.EMPTY.value:
                display_grid[y][x] = CellType.VISITED.value
        
        # Mark agent positions
        if agents:
            for agent in agents:
                if agent.agent_id in self.agent_positions:
                    x, y = agent.position
                    if 0 <= x < self.width and 0 <= y < self.height:
                        display_grid[y][x] = CellType.AGENT.value
        
        # Convert to string
        lines = []
        for row in display_grid:
            lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def get_maze_summary(self) -> Dict[str, Any]:
        """Get summary information about the maze."""
        empty_cells = sum(1 for row in self.grid for cell in row if cell == CellType.EMPTY)
        wall_cells = sum(1 for row in self.grid for cell in row if cell == CellType.WALL)
        
        return {
            'dimensions': (self.width, self.height),
            'start_position': self.start_pos,
            'exit_position': self.exit_pos,
            'empty_cells': empty_cells,
            'wall_cells': wall_cells,
            'wall_density': wall_cells / (self.width * self.height),
            'solvable': self._verify_solvable(),
            'optimal_path_length': self._calculate_optimal_path_length()
        }
    
    def _calculate_optimal_path_length(self) -> int:
        """Calculate length of optimal path from start to exit."""
        from collections import deque
        
        queue = deque([(self.start_pos, 0)])
        visited = {self.start_pos}
        
        while queue:
            (x, y), distance = queue.popleft()
            
            if (x, y) == self.exit_pos:
                return distance
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited and 
                    self.grid[ny][nx] in [CellType.EMPTY, CellType.EXIT]):
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), distance + 1))
        
        return -1  # No path found