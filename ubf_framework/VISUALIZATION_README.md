# UBF Maze Navigation Visualization

## Overview
Real-time visualization tools for observing UBF agents navigating mazes. Two visualization modes are provided:
- **ASCII Animation** (no dependencies) - Terminal-based live animation
- **2D Matplotlib** (requires matplotlib) - Graphical visualization with trails and info panels

---

## Quick Start

### ASCII Visualization (Recommended - No Installation Needed)

```bash
# Basic usage (10x10 maze, 3 agents)
python visualize_ascii.py

# Custom configuration
python visualize_ascii.py --size 15 --agents 4 --speed 0.05

# Slower animation for observation
python visualize_ascii.py --size 12 --agents 2 --speed 0.3
```

**Parameters:**
- `--size`: Maze dimension (creates square maze, default: 12)
- `--agents`: Number of agents (default: 3)
- `--speed`: Seconds per frame (default: 0.1, lower = faster)

**Controls:**
- Press `Enter` to start
- Press `Ctrl+C` to stop early

---

### 2D Matplotlib Visualization (Requires Installation)

**Installation:**
```bash
pip install matplotlib numpy
```

**Usage:**
```python
python visualize_2d.py
```

Or programmatically:
```python
from visualize_2d import run_visual_test

# Run with default settings
run_visual_test(maze_size=(12, 12), num_agents=3)

# Custom agent configurations
agent_configs = [
    {'id': 'explorer', 'frequency': 12.0, 'coherence': 0.8},
    {'id': 'cautious', 'frequency': 5.0, 'coherence': 0.9},
]
run_visual_test(maze_size=(15, 15), agent_configs=agent_configs)
```

---

## Visualization Features

### ASCII Visualization
- **Real-time terminal animation** - Watch agents navigate step-by-step
- **Agent direction indicators** - Arrows show which way agents are facing (`^>v<`)
- **Path tracking** - Visited cells marked with `·`
- **Live statistics** - Position, distance to goal, consciousness metrics
- **Completion summary** - Success rate and individual agent results

**Display Legend:**
- `S` - Start position
- `E` - Exit position  
- `█` - Wall
- `·` - Visited cell
- `^>v<` - Agents (arrow shows direction)

### 2D Matplotlib Visualization
- **Colorful maze rendering** - Clear visual distinction of cell types
- **Agent trails** - Dashed lines show paths taken
- **Direction indicators** - Arrows on agent icons
- **Live info panel** - Real-time consciousness metrics
- **Legend** - Color-coded agents and cell types
- **Smooth animation** - Configurable frame rate

**Color Scheme:**
- Dark blue-gray: Walls
- Light gray: Empty cells
- Green: Start position
- Red: Exit position
- Lighter gray: Visited cells
- Purple/Orange/Turquoise: Agents (different colors per agent)

---

## Agent Consciousness Profiles

The visualizations include agents with different consciousness configurations:

| Agent ID | Frequency | Coherence | Behavioral Traits |
|----------|-----------|-----------|-------------------|
| balanced | 7.5 Hz | 0.70 | Moderate energy, balanced approach |
| energetic | 12.0 Hz | 0.80 | High energy, risk-taking, exploratory |
| calm | 5.0 Hz | 0.90 | Low energy, cautious, deliberate |
| focused | 9.0 Hz | 0.85 | High focus, goal-oriented |
| scattered | 4.0 Hz | 0.40 | Low coherence, unpredictable |

These different profiles lead to observable behavioral differences in navigation strategies!

---

## Example Output

### ASCII Animation Frame
```
============================================================
  UBF MAZE NAVIGATION - LIVE SIMULATION
============================================================

█ █ █ █ █ █ █ █ █ █
█ S · ·   █     █ █
█ █ █ █   █   █   █
█   ·     █   █   █
█   █ █ █ █ █ █   █
█ · > · · · █     █
█ █ █ █ █ · █ █ █ █
█ ·   ·   ·       █
█ · █ █ █ █ █ █ · █
█   █       █   · E
█ █ █ █ █ █ █ █ █ █

------------------------------------------------------------
Step: 145/500
Agents Finished: 0/2

> balanced     | Running      | Pos: (3, 5) | Dist: 8  | Freq: 7.52Hz | Coh: 0.72
v energetic    | Running      | Pos: (7, 2) | Dist: 9  | Freq: 12.1Hz | Coh: 0.81

Legend: S=Start E=Exit ·=Visited █=Wall ^>v<=Agents
------------------------------------------------------------
```

### Completion Summary
```
============================================================
  SIMULATION COMPLETE
============================================================

Steps Executed: 287
Success Rate: 100.0% (2/2 agents)

Agent Results:
  ✓ balanced: SUCCESS
  ✓ energetic: SUCCESS

============================================================
```

---

## Performance Tips

### For Faster Visualization
```bash
python visualize_ascii.py --speed 0.01  # Very fast
```

### For Detailed Observation
```bash
python visualize_ascii.py --speed 0.5   # Slower, easier to watch
```

### For Large Mazes
```bash
# ASCII works well up to ~20x20
python visualize_ascii.py --size 20 --agents 5 --speed 0.05

# Matplotlib better for larger mazes
python visualize_2d.py  # Adjust maze_size in code
```

---

## Technical Details

### Frame Rate
- **ASCII**: Configurable via `--speed` parameter (default 0.1s = 10 FPS)
- **Matplotlib**: Set via `update_interval` (default 50ms = 20 FPS)

### Terminal Requirements (ASCII)
- Windows: Command Prompt or PowerShell
- Linux/Mac: Any terminal with ANSI support
- Minimum size: 80x40 characters recommended

### Display Update Strategy
Both visualizers use the environment's `step()` method to advance simulation, ensuring synchronized updates between visualization and agent decision-making.

---

## Integration with Main Framework

The visualizers work seamlessly with the existing UBF framework:

```python
from simulation.maze_environment import MazeEnvironment
from core.agent import Agent
from visualize_ascii import ASCIIMazeVisualizer

# Create environment and agents (same as main.py)
env = MazeEnvironment(12, 12)
agents = [Agent('test_agent', 7.5, 0.7)]
for agent in agents:
    env.add_agent(agent)

# Run visualization
viz = ASCIIMazeVisualizer(env, agents, update_delay=0.1)
viz.run()
```

---

## Troubleshooting

**Issue: Screen flickers**
- Increase `--speed` to slow down the animation
- Some terminals handle clearing better than others

**Issue: Agents not moving**
- This is expected sometimes - the decision system may select turning actions
- Watch for several steps to see movement patterns

**Issue: Matplotlib not found**
- Install with: `pip install matplotlib numpy`
- Or use ASCII visualization instead (no dependencies)

**Issue: Terminal too small**
- Resize terminal window
- Use smaller maze: `--size 8`

---

## Future Enhancements

Planned features:
- [ ] Pause/resume control
- [ ] Step-by-step mode (manual advance)
- [ ] Heatmap of most visited cells
- [ ] Decision reasoning overlay
- [ ] Consciousness trajectory graphs
- [ ] Multi-maze comparison view
- [ ] Export animation to GIF/video

---

*Part of the Universal Behavioral Framework (UBF) project*
