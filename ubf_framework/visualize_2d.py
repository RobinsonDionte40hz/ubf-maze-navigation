"""
2D Visualization for UBF Maze Navigation
Uses matplotlib for real-time rendering of agent movement through mazes.

OPTIONAL DEPENDENCIES:
This module requires matplotlib and numpy to be installed:
    pip install matplotlib numpy

If you don't want to install these, use visualize_ascii.py instead,
which provides terminal-based visualization with no dependencies.
"""

import sys

# Try to import optional dependencies with helpful error message
# type: ignore - these are optional dependencies
try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as patches  # type: ignore
    from matplotlib.animation import FuncAnimation  # type: ignore
    import numpy as np  # type: ignore
except ImportError as e:
    print("=" * 70)
    print("ERROR: Optional dependencies not installed")
    print("=" * 70)
    print("\nThis visualization requires matplotlib and numpy.")
    print("\nTo install:")
    print("  pip install matplotlib numpy")
    print("\nAlternatively, use the ASCII visualization (no dependencies):")
    print("  python visualize_ascii.py")
    print("\n" + "=" * 70)
    sys.exit(1)

from typing import List, Dict, Any, Tuple, Optional
import sys
sys.path.insert(0, '.')

from simulation.maze_environment import MazeEnvironment, CellType
from core.agent import Agent


class MazeVisualizer:
    """Real-time 2D visualization of maze navigation."""
    
    def __init__(self, maze_env: MazeEnvironment, agents: List[Agent], 
                 cell_size: int = 50, update_interval: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            maze_env: The maze environment to visualize
            agents: List of agents to track
            cell_size: Size of each cell in pixels
            update_interval: Milliseconds between animation frames
        """
        self.env = maze_env
        self.agents = agents
        self.cell_size = cell_size
        self.update_interval = update_interval
        
        # Color scheme
        self.colors = {
            'wall': '#2C3E50',      # Dark blue-gray
            'empty': '#ECF0F1',     # Light gray
            'start': '#2ECC71',     # Green
            'exit': '#E74C3C',      # Red
            'visited': '#D5DBDB',   # Lighter gray
            'path': '#3498DB',      # Blue
        }
        
        self.agent_colors = [
            '#9B59B6',  # Purple
            '#F39C12',  # Orange
            '#1ABC9C',  # Turquoise
            '#E67E22',  # Carrot
            '#16A085',  # Green sea
        ]
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
        
        # Animation state
        self.step_count = 0
        self.max_steps = 500
        self.running = True
        self.agent_trails = {agent.agent_id: [] for agent in agents}
        
    def setup_plot(self):
        """Initialize the plot layout."""
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Y increases downward
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Title
        self.fig.suptitle('UBF Maze Navigation Visualization', 
                         fontsize=16, fontweight='bold')
        
        # Add legend
        self._create_legend()
        
    def _create_legend(self):
        """Create legend for visualization elements."""
        legend_elements = [
            patches.Patch(facecolor=self.colors['wall'], label='Wall'),
            patches.Patch(facecolor=self.colors['empty'], label='Empty'),
            patches.Patch(facecolor=self.colors['start'], label='Start'),
            patches.Patch(facecolor=self.colors['exit'], label='Exit'),
            patches.Patch(facecolor=self.colors['visited'], label='Visited'),
        ]
        
        for i, agent in enumerate(self.agents):
            color = self.agent_colors[i % len(self.agent_colors)]
            legend_elements.append(
                patches.Patch(facecolor=color, label=agent.agent_id)
            )
        
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(1.02, 1), fontsize=10)
        
    def render_maze(self):
        """Render the current state of the maze."""
        self.ax.clear()
        self.setup_plot()
        
        # Draw grid
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = self.env.grid[y][x]
                
                # Determine color
                if cell == CellType.WALL:
                    color = self.colors['wall']
                elif (x, y) == self.env.start_pos:
                    color = self.colors['start']
                elif (x, y) == self.env.exit_pos:
                    color = self.colors['exit']
                elif (x, y) in self.env.visited_cells:
                    color = self.colors['visited']
                else:
                    color = self.colors['empty']
                
                # Draw cell
                rect = patches.Rectangle((x, y), 1, 1, 
                                        linewidth=0.5, 
                                        edgecolor='gray',
                                        facecolor=color)
                self.ax.add_patch(rect)
        
        # Draw agent trails
        for agent_id, trail in self.agent_trails.items():
            if len(trail) > 1:
                agent_idx = next(i for i, a in enumerate(self.agents) 
                               if a.agent_id == agent_id)
                color = self.agent_colors[agent_idx % len(self.agent_colors)]
                
                xs = [p[0] + 0.5 for p in trail]
                ys = [p[1] + 0.5 for p in trail]
                self.ax.plot(xs, ys, color=color, alpha=0.3, 
                           linewidth=2, linestyle='--')
        
        # Draw agents
        for i, agent in enumerate(self.agents):
            if agent.agent_id not in self.env.agents_finished:
                x, y = agent.position
                color = self.agent_colors[i % len(self.agent_colors)]
                
                # Agent body (circle)
                circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                                      facecolor=color, 
                                      edgecolor='black', 
                                      linewidth=2,
                                      zorder=10)
                self.ax.add_patch(circle)
                
                # Direction indicator (triangle)
                direction_vectors = [(0, -0.4), (0.4, 0), (0, 0.4), (-0.4, 0)]
                dx, dy = direction_vectors[agent.orientation]
                arrow = patches.FancyArrow(x + 0.5, y + 0.5, dx, dy,
                                          width=0.1, head_width=0.2,
                                          head_length=0.15, fc='black',
                                          ec='black', zorder=11)
                self.ax.add_patch(arrow)
        
        # Add info panel
        self._add_info_panel()
        
    def _add_info_panel(self):
        """Add information panel to the plot."""
        info_text = f"Step: {self.step_count}/{self.max_steps}\n"
        info_text += f"Finished: {len(self.env.agents_finished)}/{len(self.agents)}\n\n"
        
        for i, agent in enumerate(self.agents):
            status = "✓ DONE" if agent.agent_id in self.env.agents_finished else "Running"
            info_text += f"{agent.agent_id}:\n"
            info_text += f"  Status: {status}\n"
            info_text += f"  Pos: {agent.position}\n"
            info_text += f"  Freq: {agent.consciousness.frequency:.2f} Hz\n"
            info_text += f"  Coh: {agent.consciousness.coherence:.2f}\n\n"
        
        # Place text outside the plot area
        self.fig.text(0.02, 0.5, info_text, fontsize=9, 
                     family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
    def step_simulation(self, frame):
        """Execute one simulation step and update visualization."""
        if not self.running:
            return
        
        if self.step_count >= self.max_steps:
            self.running = False
            self._show_completion_message()
            return
        
        # Check if all agents finished
        if len(self.env.agents_finished) == len(self.agents):
            self.running = False
            self._show_completion_message()
            return
        
        # Execute step
        step_result = self.env.step(self.agents)
        self.step_count += 1
        
        # Update trails
        for agent in self.agents:
            if agent.agent_id not in self.env.agents_finished:
                self.agent_trails[agent.agent_id].append(agent.position)
        
        # Render
        self.render_maze()
        
    def _show_completion_message(self):
        """Display completion message."""
        success_count = len(self.env.agents_finished)
        success_rate = (success_count / len(self.agents)) * 100
        
        msg = f"\n{'='*50}\n"
        msg += f"SIMULATION COMPLETE\n"
        msg += f"{'='*50}\n"
        msg += f"Steps: {self.step_count}\n"
        msg += f"Success Rate: {success_rate:.1f}% ({success_count}/{len(self.agents)})\n"
        msg += f"{'='*50}\n"
        
        self.ax.text(self.env.width / 2, self.env.height / 2, 
                    f"COMPLETE\n{success_count}/{len(self.agents)} Agents Finished",
                    fontsize=20, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        print(msg)
        
    def run(self):
        """Start the visualization animation."""
        print("Starting UBF Maze Navigation Visualization...")
        print(f"Maze: {self.env.width}x{self.env.height}")
        print(f"Agents: {len(self.agents)}")
        print(f"Max Steps: {self.max_steps}")
        print("\nClose the window to stop the simulation.\n")
        
        # Initial render
        self.render_maze()
        
        # Create animation
        anim = FuncAnimation(self.fig, self.step_simulation, 
                           interval=self.update_interval,
                           repeat=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def save_snapshot(self, filename: str):
        """Save current state as image."""
        self.render_maze()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Snapshot saved to {filename}")


def run_visual_test(maze_size: Tuple[int, int] = (12, 12), 
                   num_agents: int = 10,
                   agent_configs: Optional[List[Dict[str, float]]] = None):
    """
    Run a visual test of the UBF maze navigation.
    
    Args:
        maze_size: Size of the maze (width, height)
        num_agents: Number of agents to create
        agent_configs: Optional list of agent configurations
    """
    # Create environment
    env = MazeEnvironment(maze_size[0], maze_size[1])
    
    # Create agents
    agents = []
    if agent_configs:
        for i, config in enumerate(agent_configs):
            agent = Agent(
                agent_id=config.get('id', f'agent_{i}'),
                initial_frequency=config.get('frequency', 7.5),
                initial_coherence=config.get('coherence', 0.7),
                temperature=config.get('temperature', 1.0)
            )
            agents.append(agent)
    else:
        # Default agents with varied consciousness
        configs = [
            {'id': 'balanced', 'frequency': 7.5, 'coherence': 0.7},
            {'id': 'energetic', 'frequency': 12.0, 'coherence': 0.8},
            {'id': 'calm', 'frequency': 5.0, 'coherence': 0.9},
        ]
        for i in range(min(num_agents, len(configs))):
            agent = Agent(**configs[i])
            agents.append(agent)
        
        # Add more agents if needed
        for i in range(len(configs), num_agents):
            agent = Agent(f'agent_{i}', 7.5, 0.7)
            agents.append(agent)
    
    # Add agents to environment
    for agent in agents:
        env.add_agent(agent)
    
    # Create visualizer
    viz = MazeVisualizer(env, agents, cell_size=50, update_interval=50)
    
    # Run visualization
    viz.run()


if __name__ == "__main__":
    print("UBF 2D Maze Navigation Visualization")
    print("=" * 50)
    
    # Run with 10 agents
    run_visual_test(maze_size=(10, 10), num_agents=10)
