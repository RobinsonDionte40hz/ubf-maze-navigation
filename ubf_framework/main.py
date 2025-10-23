"""
Universal Behavioral Framework - Main Demo

Demonstrates the complete UBF system with maze navigation scenarios
showing learning from failures and creative adaptation.
"""

import json
import time
import random
from pathlib import Path

# Import UBF components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_scenarios import run_ubf_test
from core.agent import Agent
from simulation.maze_environment import MazeEnvironment
from core.consciousness_state import ConsciousnessState
from core.decision_system import ActionType


def demonstrate_basic_ubf():
    """Demonstrate basic UBF functionality with a single agent."""
    print("=== UBF Basic Demonstration ===")
    
    # Create agent with initial consciousness coordinates
    agent = Agent("demo_agent", initial_frequency=7.5, initial_coherence=0.7)
    print(f"Initial agent state: {agent}")
    print(f"Behavioral state: {agent.behavioral_state}")
    
    # Create maze environment
    maze = MazeEnvironment(width=10, height=10)
    maze.add_agent(agent)
    
    print(f"\nMaze info: {maze.get_maze_summary()}")
    print("\nInitial maze:")
    print(maze.render_ascii([agent]))
    
    # Run a few steps to show decision making
    print(f"\n--- Running 10 steps ---")
    for step in range(10):
        # Get environmental context
        env_context = maze._get_environmental_context(agent)
        
        # Agent selects action
        selected_action, decision_breakdown = agent.select_action(env_context)
        
        # Execute action
        action_result = maze.execute_agent_action(agent, selected_action.action_type)
        
        # Store action before processing outcome
        agent.last_action = selected_action
        
        # Process outcome
        agent.process_outcome(action_result)
        
        # Print step info
        print(f"Step {step + 1}: {selected_action.action_type.value} → {action_result['outcome']} (reward: {action_result['reward']:.3f})")
        print(f"  Position: {agent.position}, Consciousness: {agent.consciousness.frequency:.2f}Hz/{agent.consciousness.coherence:.2f}")
        
        if agent.goal_achieved:
            print("Goal achieved!")
            break
    
    print(f"\nFinal agent state: {agent}")
    print(f"Memory count: {len(agent.memory_manager.memories)}")
    
    if agent.memory_manager.memories:
        print("\nRecent memories:")
        for memory in agent.memory_manager.memories[-3:]:
            print(f"  {memory.description} (sig: {memory.significance:.2f}, emotional: {memory.emotional_impact:+.2f})")


def demonstrate_failure_learning():
    """Demonstrate how agents learn from failures and adapt creatively."""
    print("\n=== Failure Learning Demonstration ===")
    
    # Create agent with moderate consciousness
    agent = Agent("failure_learner", initial_frequency=6.0, initial_coherence=0.5)
    
    # Create maze with more walls to force failures
    maze = MazeEnvironment(width=8, height=8, wall_density=0.4)
    maze.add_agent(agent)
    
    print(f"Starting consciousness: {agent.consciousness}")
    print(f"Starting behavioral state - creativity: {agent.behavioral_state.creativity:.2f}, risk_tolerance: {agent.behavioral_state.risk_tolerance:.2f}")
    
    failures = []
    creative_responses = []
    
    # Run simulation and track failures/responses
    for step in range(30):
        env_context = maze._get_environmental_context(agent)
        selected_action, decision_breakdown = agent.select_action(env_context)
        action_result = maze.execute_agent_action(agent, selected_action.action_type)
        
        # Track failures and subsequent creative responses
        if action_result['outcome'] in ['collision', 'failure']:
            failures.append({
                'step': step,
                'action': selected_action.action_type.value,
                'consciousness_before': (agent.consciousness.frequency, agent.consciousness.coherence)
            })
            print(f"Step {step + 1}: FAILURE - {selected_action.action_type.value} → {action_result['outcome']}")
        
        agent.process_outcome(action_result)
        
        # Check for creative responses after failures
        if (failures and selected_action.action_type in [ActionType.CREATIVE_SOLUTION, ActionType.INVESTIGATE] and
            step - failures[-1]['step'] <= 2):
            creative_responses.append({
                'step': step,
                'action': selected_action.action_type.value,
                'response_to_failure': failures[-1]
            })
            print(f"Step {step + 1}: CREATIVE RESPONSE - {selected_action.action_type.value}")
        
        if agent.goal_achieved:
            break
    
    print(f"\nFinal consciousness: {agent.consciousness}")
    print(f"Final behavioral state - creativity: {agent.behavioral_state.creativity:.2f}, risk_tolerance: {agent.behavioral_state.risk_tolerance:.2f}")
    print(f"Total failures: {len(failures)}")
    print(f"Creative responses: {len(creative_responses)}")
    
    # Show consciousness evolution
    if len(agent.action_history) > 1:
        start_freq = agent.action_history[0]['consciousness']['frequency']
        end_freq = agent.action_history[-1]['consciousness']['frequency']
        start_coh = agent.action_history[0]['consciousness']['coherence']
        end_coh = agent.action_history[-1]['consciousness']['coherence']
        
        print(f"Consciousness evolution:")
        print(f"  Frequency: {start_freq:.2f} → {end_freq:.2f} (Δ{end_freq-start_freq:+.2f})")
        print(f"  Coherence: {start_coh:.2f} → {end_coh:.2f} (Δ{end_coh-start_coh:+.2f})")


def run_full_test_suite():
    """Run the complete test suite with all scenarios."""
    print("\n=== Full UBF Test Suite ===")
    
    # Set random seed for reproducibility
    random_seed = 12345
    
    # Run all test scenarios
    results = run_ubf_test(
        maze_size=(10, 10),
        max_steps=500,
        random_seed=random_seed,
        save_file="ubf_test_results.json"
    )
    
    # Print key findings
    print("\n=== Key Findings ===")
    
    analysis = results['analysis']
    
    # Learning progression
    if 'learning_progression' in analysis:
        progression = analysis['learning_progression']
        print(f"\nLearning Progression:")
        print(f"  Time improvement: {progression.get('time_improvement_pct', 0):.1f}%")
        print(f"  Learning trend: {progression.get('learning_trend', 'unknown')}")
    
    # Failure adaptation
    if 'failure_adaptation' in analysis:
        adaptation = analysis['failure_adaptation']
        print(f"\nFailure Adaptation:")
        print(f"  Agents with adaptations: {adaptation.get('agents_with_adaptations', 0)}")
        print(f"  Total adaptations: {adaptation.get('total_failure_adaptations', 0)}")
        print(f"  Common responses: {list(adaptation.get('common_adaptations', {}).keys())[:3]}")
    
    # Memory patterns
    if 'memory_patterns' in analysis:
        memory = analysis['memory_patterns']
        print(f"\nMemory Formation:")
        print(f"  Total memories: {memory.get('total_memories_formed', 0)}")
        print(f"  Average significance: {memory.get('average_significance', 0):.3f}")
        print(f"  Agents with memories: {memory.get('agents_with_memories', 0)}")
    
    # Performance comparison
    if 'performance_comparison' in analysis:
        performance = analysis['performance_comparison']
        print(f"\nPerformance Comparison:")
        for scenario, metrics in performance.items():
            print(f"  {scenario}: {metrics.get('completion_rate', 0):.1%} success, {metrics.get('average_steps', 0):.0f} steps")
    
    return results


def demonstrate_consciousness_coordinates():
    """Demonstrate how different consciousness coordinates affect behavior."""
    print("\n=== Consciousness Coordinates Demo ===")
    
    # Create agents with different consciousness states
    agents = [
        Agent("low_energy", initial_frequency=4.0, initial_coherence=0.6),      # Lethargic, moderate focus
        Agent("high_energy", initial_frequency=12.0, initial_coherence=0.6),    # Energetic, moderate focus
        Agent("scattered", initial_frequency=8.0, initial_coherence=0.3),       # Moderate energy, scattered
        Agent("focused", initial_frequency=8.0, initial_coherence=0.9),         # Moderate energy, focused
    ]
    
    print("Agent behavioral profiles:")
    for agent in agents:
        bs = agent.behavioral_state
        print(f"{agent.agent_id:12}: energy={bs.energy:.2f}, focus={bs.focus:.2f}, mood={bs.mood:+.2f}, "
              f"social={bs.social_drive:.2f}, risk={bs.risk_tolerance:.2f}, creativity={bs.creativity:.2f}")
    
    # Test action preferences in identical situation
    maze = MazeEnvironment(6, 6)
    test_context = {
        'obstacles_ahead': True,
        'unexplored_areas': True,
        'terrain_difficulty': 0.5,
        'goal_progress': 0.3,
        'novelty_factor': 0.4
    }
    
    print(f"\nAction preferences in identical situation:")
    for agent in agents:
        actions = agent.get_available_actions(test_context)
        weights = []
        
        for action in actions:
            weight, factors = agent.decision_system.calculate_interaction_weight(
                consciousness=agent.consciousness,
                behavioral_state=agent.behavioral_state,
                memory_manager=agent.memory_manager,
                action=action,
                environmental_context=test_context
            )
            weights.append((action.action_type.value, weight))
        
        # Sort by weight and show top 3
        weights.sort(key=lambda x: x[1], reverse=True)
        top_actions = [f"{act}({wt:.2f})" for act, wt in weights[:3]]
        print(f"{agent.agent_id:12}: {', '.join(top_actions)}")


def main():
    """Main demonstration function."""
    print("Universal Behavioral Framework (UBF) - Complete Demonstration")
    print("=" * 60)
    
    # Basic demonstration
    demonstrate_basic_ubf()
    
    # Consciousness coordinates demo
    demonstrate_consciousness_coordinates()
    
    # Failure learning demo
    demonstrate_failure_learning()
    
    # Full test suite
    results = run_full_test_suite()
    
    print("\n" + "=" * 60)
    print("UBF Demonstration Complete!")
    print(f"Full results saved to: ubf_test_results.json")
    print(f"Data size: ~{len(json.dumps(results)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()