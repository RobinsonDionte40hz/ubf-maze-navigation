"""
Debug Memory Pipeline Test
"""

from core.agent import Agent
from simulation.maze_environment import MazeEnvironment
from core.decision_system import ActionType

print('=== DEBUGGING MEMORY PIPELINE ===')

# Create agent
agent = Agent('debug_agent', initial_frequency=7.5, initial_coherence=0.7)
maze = MazeEnvironment(width=6, height=6)
maze.add_agent(agent)

print(f'Initial memory count: {len(agent.memory_manager.memories)}')

# Force a collision by trying to move forward from start position (should hit wall)
print('Forcing a collision...')
action_result = {
    'action': 'move_forward',
    'success': False,
    'outcome': 'collision',
    'reward': -0.5,
    'position_change': (0, 0),
    'new_information': False,
    'surprise': 0.8,
    'context': {'old_position': agent.position}
}

# Set last action manually
from core.decision_system import Action
agent.last_action = Action(ActionType.MOVE_FORWARD, goal_alignment=0.5, interaction_type='exploration')

# Process the outcome
agent.process_outcome(action_result)

print(f'Final memory count: {len(agent.memory_manager.memories)}')
print('Test completed')