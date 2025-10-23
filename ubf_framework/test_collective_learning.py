"""
Test script for collective memory and respawn functionality.
Demonstrates agents learning from each other's successes.
"""

import sys
sys.path.insert(0, '.')

from simulation.maze_environment import MazeEnvironment
from core.agent import Agent
from core.collective_memory import CollectiveMemoryPool


def run_collective_learning_test(maze_size=(12, 12), num_agents=10, 
                                 max_steps_per_run=100, respawn_limit=3):
    """
    Test collective learning where agents share knowledge and respawn.
    
    Args:
        maze_size: Size of the maze
        num_agents: Number of agents
        max_steps_per_run: Max steps before respawn
        respawn_limit: Maximum number of respawns per agent
    """
    print("=" * 70)
    print("  COLLECTIVE LEARNING TEST")
    print("  Agents share successful strategies and respawn to try again")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Maze Size: {maze_size[0]}x{maze_size[1]}")
    print(f"  Agents: {num_agents}")
    print(f"  Max Steps Per Life: {max_steps_per_run if max_steps_per_run else 'Unlimited'}")
    print(f"  Respawn Limit: {respawn_limit}")
    print()
    
    # Create environment (no hard step limit!)
    env = MazeEnvironment(maze_size[0], maze_size[1], max_steps=None)
    
    # Create collective memory pool (shared by all agents)
    collective_memory = CollectiveMemoryPool()
    
    # Create agents with collective memory
    agents = []
    for i in range(num_agents):
        agent = Agent(
            agent_id=f"agent_{i}",
            initial_frequency=7.5 + (i * 0.2),  # Slight variation
            initial_coherence=0.7,
            temperature=1.0,
            collective_memory=collective_memory  # Shared pool!
        )
        agents.append(agent)
        env.add_agent(agent)
    
    print(f"Created {len(agents)} agents with shared collective memory pool")
    print()
    
    # Statistics tracking
    generation_stats = []
    total_successes = 0
    total_respawns = 0
    
    # Run simulation with continuous respawning every 25 steps
    total_step_count = 0
    max_total_steps = max_steps_per_run * respawn_limit  # Overall limit to prevent infinite loops
    respawn_interval = 25  # Respawn every 25 steps
    
    print(f"Running continuous learning (respawn every {respawn_interval} steps)...\n")
    
    while total_step_count < max_total_steps:
        collective_before = len(collective_memory.memories)
        
        # Run for respawn_interval steps
        for step in range(respawn_interval):
            total_step_count += 1
            
            for agent in agents:
                # Skip agents that have succeeded or exhausted respawns
                if agent.goal_achieved or agent.respawn_count >= respawn_limit:
                    continue
                
                # Take action
                context = env.get_environmental_context(agent)
                action, _ = agent.select_action(context)
                result = env.execute_agent_action(agent, action.action_type)
                agent.process_outcome(result)  # Broadcasts real-time!
                
                # Check if reached goal
                if result.get('outcome') == 'goal_achieved':
                    print(f"  ✓ {agent.agent_id} REACHED EXIT at step {total_step_count}! (personal steps: {agent.step_count}, respawns: {agent.respawn_count})")
                    total_successes += 1
            
            # Every 50 global steps, show progress
            if total_step_count % 50 == 0:
                collective_after = len(collective_memory.memories)
                active_count = sum(1 for a in agents if not a.goal_achieved and a.respawn_count < respawn_limit)
                print(f"  [Global Step {total_step_count}] Active: {active_count}, Successes: {total_successes}, Collective Memories: {collective_after} (+{collective_after - collective_before})")
                collective_before = collective_after
        
        # RESPAWN CYCLE: Every 25 steps, respawn agents that haven't succeeded
        respawned_this_cycle = 0
        for agent in agents:
            if not agent.goal_achieved and agent.respawn_count < respawn_limit:
                env.respawn_agent(agent)  # Use environment method!
                total_respawns += 1
                respawned_this_cycle += 1
        
        if respawned_this_cycle > 0:
            print(f"  → Respawned {respawned_this_cycle} agents at step {total_step_count} (back to start position)")
        
        # Check if all agents have either succeeded or exhausted respawns
        active_count = sum(1 for a in agents if not a.goal_achieved and a.respawn_count < respawn_limit)
        if active_count == 0:
            print(f"\n✓ All agents complete at step {total_step_count}!")
            break
    
    # Final statistics
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\nTotal Steps: {total_step_count}")
    print(f"Total Successes: {total_successes}/{num_agents} ({total_successes/num_agents*100:.1f}%)")
    print(f"Total Respawns: {total_respawns}")
    print(f"Collective Memory Pool: {len(collective_memory.memories)} shared memories")
    print()
    
    stats = collective_memory.get_statistics()
    if stats['total_memories'] > 0:
        print(f"Collective Knowledge Stats:")
        print(f"  Average Reliability: {stats['avg_reliability']:.2f}")
        print(f"  Total Contributors: {stats['contributors']}")
        print(f"  Most Reliable Pattern: {stats['most_reliable_pattern']['location']}")
        print(f"    → Reliability: {stats['most_reliable_pattern']['reliability']:.2f}")
        print(f"    → Contributors: {stats['most_reliable_pattern']['contributors']}")
        print()
    
    # Show agent-by-agent results
    print("Individual Agent Results:")
    for agent in agents:
        status = "✓ SUCCESS" if agent.goal_achieved else "✗ Failed"
        print(f"  {agent.agent_id}: {status} (respawns: {agent.respawn_count}, "
              f"personal memories: {len(agent.memory_manager.memories)})")
    
    print("\n" + "=" * 70)
    
    return {
        'success_rate': total_successes / num_agents,
        'total_steps': total_step_count,
        'total_respawns': total_respawns,
        'collective_memories': len(collective_memory.memories)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test collective learning and respawn')
    parser.add_argument('--size', type=int, default=12, help='Maze size')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--steps', type=int, default=100, help='Max steps per life')
    parser.add_argument('--respawns', type=int, default=3, help='Respawn limit per agent')
    
    args = parser.parse_args()
    
    results = run_collective_learning_test(
        maze_size=(args.size, args.size),
        num_agents=args.agents,
        max_steps_per_run=args.steps,
        respawn_limit=args.respawns
    )
