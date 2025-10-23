from core.agent import Agent
from core.decision_system import ActionType
from simulation.maze_environment import MazeEnvironment

# Create a maze where agent will definitely hit a wall
env = MazeEnvironment(width=5, height=5, max_steps=10)

# Create agent
from core.collective_memory import CollectiveMemoryPool
collective_memory = CollectiveMemoryPool()
agent = Agent('test_agent', collective_memory=collective_memory)
env.add_agent(agent)

print('Maze layout:')
print(env.render_ascii([agent]))

# Force the agent to try moving in a direction that will hit a wall
agent.orientation = 0  # Face north

# Need to set last_action for process_outcome to work
from core.decision_system import Action
from core.memory_system import InteractionType
agent.last_action = Action(
    action_type=ActionType.MOVE_FORWARD,
    interaction_type=InteractionType.EXPLORATION,
    goal_alignment=0.6
)

print(f'Agent at {agent.position}, facing {agent.orientation} (0=North)')

# Try to move forward - this should hit a wall
action_result = env.execute_agent_action(agent, ActionType.MOVE_FORWARD)
print(f'Action result: {action_result}')

# Process the outcome
agent.process_outcome(action_result)

print(f'Agent memories after collision: {len(agent.memory_manager.memories)}')
print(f'Collective memories: {len(agent.collective_memory.memories) if agent.collective_memory else 0}')

# Check memory stats
if agent.memory_manager.memories:
    mem = agent.memory_manager.memories[0]
    print(f'Memory details: significance={mem.significance}, emotional_impact={mem.emotional_impact}, outcome={mem.outcome}')