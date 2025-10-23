"""
ASCII Animation for UBF Maze Navigation
Terminal-based real-time visualization without external dependencies.
"""

import sys
import time
import os
from typing import List, Dict, Any
sys.path.insert(0, '.')

from simulation.maze_environment import MazeEnvironment, CellType
from core.agent import Agent
from core.collective_memory import CollectiveMemoryPool


class ASCIIMazeVisualizer:
    """Terminal-based real-time ASCII visualization of maze navigation."""
    
    def __init__(self, maze_env: MazeEnvironment, agents: List[Agent], 
                 update_delay: float = 0.1, collective_memory: CollectiveMemoryPool = None,
                 failures_before_respawn: int = 10):
        """
        Initialize the ASCII visualizer.
        
        Args:
            maze_env: The maze environment to visualize
            agents: List of agents to track
            update_delay: Seconds between frames
            collective_memory: Shared memory pool for group learning
            failures_before_respawn: Respawn after N failures (collisions)
        """
        self.env = maze_env
        self.agents = agents
        self.update_delay = update_delay
        self.collective_memory = collective_memory
        self.failures_before_respawn = failures_before_respawn
        self.respawn_limit = None  # No limit - let them keep trying until they solve it!
        
        # Track failures per agent
        self.agent_failure_counts = {agent.agent_id: 0 for agent in agents}
        
        # Agent symbols (different for each agent)
        self.agent_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # Direction arrows
        self.direction_symbols = {
            0: '^',  # North
            1: '>',  # East
            2: 'v',  # South
            3: '<',  # West
        }
        
        self.step_count = 0
        self.total_respawns = 0
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def render_frame(self):
        """Render the current state as ASCII."""
        # Build display grid
        display = []
        for y in range(self.env.height):
            row = []
            for x in range(self.env.width):
                cell = self.env.grid[y][x]
                
                if cell == CellType.WALL:
                    row.append('█')
                elif (x, y) == self.env.start_pos:
                    row.append('S')
                elif (x, y) == self.env.exit_pos:
                    row.append('E')
                elif (x, y) in self.env.visited_cells:
                    row.append('·')
                else:
                    row.append(' ')
            display.append(row)
        
        # Overlay agents
        for i, agent in enumerate(self.agents):
            if agent.agent_id not in self.env.agents_finished:
                x, y = agent.position
                if 0 <= x < self.env.width and 0 <= y < self.env.height:
                    # Use directional arrow
                    display[y][x] = self.direction_symbols[agent.orientation]
        
        # Clear and print
        self.clear_screen()
        
        # Header
        print("=" * 60)
        print("  UBF MAZE NAVIGATION - LIVE SIMULATION")
        print("=" * 60)
        print()
        
        # Maze
        for row in display:
            print(' '.join(row))
        
        print()
        print("-" * 60)
        
        # Info panel
        print(f"Step: {self.step_count}")
        print(f"Agents Finished: {len(self.env.agents_finished)}/{len(self.agents)} | Total Respawns: {self.total_respawns}")
        if self.collective_memory:
            print(f"Collective Memories: {len(self.collective_memory.memories)}")
        print()
        
        # Agent status
        for i, agent in enumerate(self.agents):
            symbol = self.direction_symbols[agent.orientation]
            status = "✓ FINISHED" if agent.agent_id in self.env.agents_finished else "Running"
            pos = agent.position
            dist = abs(pos[0] - self.env.exit_pos[0]) + abs(pos[1] - self.env.exit_pos[1])
            failures = self.agent_failure_counts[agent.agent_id]
            
            print(f"{symbol} {agent.agent_id:12s} | {status:12s} | Respawns: {agent.respawn_count} | "
                  f"Failures: {failures}/{self.failures_before_respawn} | Pos: {pos} | Dist: {dist:2d}")
        
        print()
        print("Legend: S=Start E=Exit ·=Visited █=Wall ^>v<=Agents")
        print(f"Respawning after {self.failures_before_respawn} failures (collisions)")
        print("-" * 60)
        
    def run(self):
        """Run the ASCII animation with failure-based respawning."""
        print("\nStarting ASCII visualization with collective learning...")
        print(f"Agents will respawn after {self.failures_before_respawn} failures (collisions)")
        print("Press Ctrl+C to stop\n")
        time.sleep(1)
        
        try:
            while True:
                # Check if all agents succeeded (no respawn limit!)
                active_count = sum(1 for a in self.agents if not a.goal_achieved)
                if active_count == 0:
                    self.render_frame()
                    self._show_completion()
                    break
                
                # Step simulation and track failures
                step_results = self.env.step(self.agents)
                self.step_count += 1
                
                # Check for failures (collisions) and respawn if threshold reached
                for agent_id, result in step_results['agent_results'].items():
                    agent = next(a for a in self.agents if a.agent_id == agent_id)
                    
                    # Count failures (collisions with walls)
                    if result['outcome'] == 'collision':
                        self.agent_failure_counts[agent_id] += 1
                    
                    # Respawn if failure threshold reached
                    if (self.agent_failure_counts[agent_id] >= self.failures_before_respawn 
                        and not agent.goal_achieved):
                        self.env.respawn_agent(agent)
                        self.total_respawns += 1
                        self.agent_failure_counts[agent_id] = 0  # Reset failure count
                
                # Render
                self.render_frame()
                
                # Delay
                time.sleep(self.update_delay)
                
        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user.")
            self._show_completion()
    
    def _show_completion(self):
        """Display completion summary."""
        print("\n" + "=" * 60)
        print("  SIMULATION COMPLETE")
        print("=" * 60)
        
        success_count = len(self.env.agents_finished)
        success_rate = (success_count / len(self.agents)) * 100
        
        print(f"\nSteps Executed: {self.step_count}")
        print(f"Success Rate: {success_rate:.1f}% ({success_count}/{len(self.agents)} agents)")
        print(f"\nAgent Results:")
        
        for agent in self.agents:
            if agent.agent_id in self.env.agents_finished:
                print(f"  ✓ {agent.agent_id}: SUCCESS")
            else:
                dist = abs(agent.position[0] - self.env.exit_pos[0]) + \
                       abs(agent.position[1] - self.env.exit_pos[1])
                print(f"  ✗ {agent.agent_id}: INCOMPLETE (distance to goal: {dist})")
        
        print("\n" + "=" * 60)


def run_ascii_visualization(maze_size=(12, 12), num_agents=3, speed=0.1):
    """
    Run an ASCII-based visualization with collective learning.
    
    Args:
        maze_size: Tuple of (width, height)
        num_agents: Number of agents to simulate
        speed: Delay between frames in seconds (lower = faster)
    """
    # Create environment (no step limit!)
    env = MazeEnvironment(maze_size[0], maze_size[1], max_steps=None)
    
    # Create collective memory pool
    collective_memory = CollectiveMemoryPool()
    
    # Create IDENTICAL agents - all should learn the same way!
    # No personality differences, just pure learning
    agents = []
    for i in range(num_agents):
        agent = Agent(f'agent_{i}', 
                     initial_frequency=8.0,  # All same
                     initial_coherence=0.75,  # All same
                     temperature=1.2,  # All same - moderate exploration
                     collective_memory=collective_memory)
        agents.append(agent)
    
    # Add agents to environment
    for agent in agents:
        env.add_agent(agent)
    
    # Create and run visualizer with collective memory
    viz = ASCIIMazeVisualizer(env, agents, update_delay=speed, 
                             collective_memory=collective_memory,
                             failures_before_respawn=5)  # Respawn after 5 collisions (was 10)
    viz.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='UBF Maze Navigation ASCII Visualization')
    parser.add_argument('--size', type=int, default=12, help='Maze size (creates square maze)')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--speed', type=float, default=0.1, help='Animation speed (seconds per frame)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  UBF MAZE NAVIGATION - ASCII VISUALIZATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Maze Size: {args.size}x{args.size}")
    print(f"  Agents: {args.agents}")
    print(f"  Speed: {args.speed}s per frame")
    print("\n" + "=" * 60)
    
    input("\nPress Enter to start...")
    
    run_ascii_visualization(
        maze_size=(args.size, args.size),
        num_agents=args.agents,
        speed=args.speed
    )
