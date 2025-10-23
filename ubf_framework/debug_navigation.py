"""Debug script to understand why agents aren't navigating mazes."""
import sys
sys.path.insert(0, '.')

from simulation.maze_environment import MazeEnvironment
from core.agent import Agent
import random

def debug_agent_navigation():
    """Run a simple maze navigation test with detailed logging."""
    random.seed(42)
    
    # Create small maze and agent
    env = MazeEnvironment(8, 8)
    agent = Agent('debug_agent', 7.5, 0.7)
    env.add_agent(agent)
    
    print("=" * 60)
    print("MAZE NAVIGATION DEBUG")
    print("=" * 60)
    print(f"\nMaze size: {env.width}x{env.height}")
    print(f"Start position: {env.start_pos}")
    print(f"Exit position: {env.exit_pos}")
    print(f"Agent starting at: {agent.position}, facing: {agent.orientation}")
    print(f"\nInitial maze:")
    print(env.render_ascii([agent]))
    
    # Run 20 steps with detailed logging
    print(f"\n{'='*60}")
    print("STEP-BY-STEP EXECUTION")
    print(f"{'='*60}\n")
    
    action_counts = {}
    position_history = [agent.position]
    
    for step in range(1, 21):
        # Get context
        context = env._get_environmental_context(agent)
        
        # Agent selects action
        selected_action, decision_breakdown = agent.select_action(context)
        action_type = selected_action.action_type.value
        
        # Track action frequency
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Execute in environment
        result = env.execute_agent_action(agent, selected_action.action_type)
        
        # Agent processes
        agent.process_outcome(result)
        
        # Track position
        position_history.append(agent.position)
        
        print(f"Step {step:2d}: {action_type:20s} → {result['outcome']:15s} "
              f"(reward: {result['reward']:+.3f}) | "
              f"Pos: {agent.position}, Dir: {agent.orientation}")
        
        if step % 5 == 0:
            print(f"         Visible cells: {len(context.get('visible_cells', []))}")
            print(f"         Distance to goal: {context.get('distance_to_goal', 'N/A')}")
            print()
        
        if agent.goal_achieved:
            print(f"\n🎉 GOAL ACHIEVED at step {step}!")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Goal achieved: {agent.goal_achieved}")
    print(f"Final position: {agent.position} (target: {env.exit_pos})")
    print(f"Distance to goal: {abs(agent.position[0] - env.exit_pos[0]) + abs(agent.position[1] - env.exit_pos[1])}")
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:20s}: {count:2d} ({count/sum(action_counts.values())*100:.1f}%)")
    
    # Check if agent moved at all
    unique_positions = set(position_history)
    print(f"\nMovement analysis:")
    print(f"  Unique positions visited: {len(unique_positions)}")
    print(f"  Positions: {unique_positions}")
    print(f"  Agent stuck: {len(unique_positions) == 1}")
    
    print(f"\nFinal maze state:")
    print(env.render_ascii([agent]))
    
    # Check consciousness state
    print(f"\nConsciousness evolution:")
    print(f"  Frequency: {agent.consciousness.frequency:.2f} Hz")
    print(f"  Coherence: {agent.consciousness.coherence:.2f}")
    print(f"  Behavioral state: {agent.behavioral_state}")

if __name__ == "__main__":
    debug_agent_navigation()
