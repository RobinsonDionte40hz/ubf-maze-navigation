# UBF Maze Navigation - AI Coding Guidelines

## Architecture Overview
This is the Universal Behavioral Framework (UBF) implementing consciousness-based autonomous maze navigation. The system consists of:

- **Core UBF Components** (`core/`): Consciousness coordinates (frequency/coherence), memory systems, 13-factor decision engine, event-driven updates
- **Agent Integration** (`core/agent.py`): Combines all subsystems with direct wall/path tracking
- **Maze Simulation** (`simulation/maze_environment.py`): DFS-carved mazes with triangular line-of-sight
- **Testing Framework** (`tests/test_scenarios.py`): Solo phase (3 runs) + group scenarios with experience sharing

## Key Data Flows
```
Environment Perception → Memory Retrieval → 13-Factor Weight Calculation → Softmax Action Selection → Execution → Consciousness/Memory Update
```

## Critical Workflows
- **Run Tests**: `python main.py` (core tests) or `python demo.py` (full suite + visualization)
- **Visualize**: `python visualize_ascii.py --size 12 --agents 3 --speed 0.1` (no dependencies) or `python visualize_2d.py` (requires matplotlib)
- **Debug Navigation**: `python debug_navigation.py`
- **Dependencies**: Pure Python standard library; optional matplotlib for 2D visualization

## Project Conventions

### Code Structure
- Use `dataclasses` for data models (e.g., `Memory`, `Action`)
- Extensive type hints with `typing` module
- Enums for categorical values (`InteractionType`, `ActionType`, `CellType`)
- Composition pattern: Agent combines consciousness, memory, decision systems

### Consciousness System
- **Coordinates**: Frequency (3-15 Hz), Coherence (0.2-1.0)
- **Behavioral Mapping**: 8 dimensions derived from coordinates
- **Updates**: Event-driven with noise scheduling for recovery mode

### Memory System
- **Significance Threshold**: 0.15 minimum for memory formation
- **Influence Range**: 0.3x-2.5x multiplier (avoidance to strong preference)
- **Decay**: Exponential with half-life based on significance
- **Consolidation**: Quantum-inspired merging of similar memories

### Decision System
- **13 Factors**: Goal alignment (+12.0 for forward toward goal), critical needs, environmental suitability, memory influence, consciousness modifiers, etc.
- **Action Selection**: Temperature-based softmax (0.1-2.0 range)
- **Failure Boost**: Temporary temperature increase after collisions

### Maze Navigation Specifics
- **Actions**: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT (simplified from full action set)
- **Rewards**: +0.5 for successful moves, -0.5 for collisions, +10.0 for goal completion
- **Direct Tracking**: Agent maintains `wall_positions` set and `successful_moves` dict for immediate collision avoidance
- **Smart Respawn**: Memory-based orientation selection from start position

### Testing Patterns
- **Scenarios**: Solo phase (3 runs, keep memories), experienced + new agents, all-new agents
- **Metrics**: Completion rate, step count, consciousness evolution
- **Analysis**: Learning progression, failure adaptation, memory patterns

## Integration Points
- **Collective Memory**: Shared memory pool for group learning with real-time broadcasting
- **Experience Sharing**: Post-run consciousness coordinate averaging between agents
- **Event System**: Standardized event data structure for consciousness updates

## Common Patterns
- **Memory Queries**: Check target location for MOVE_FORWARD actions, current location for turns
- **Weight Calculations**: Apply memory influence as multiplier (0.3x-2.5x range)
- **Consciousness Bounds**: Clamp frequency (3-15 Hz), coherence (0.2-1.0) after updates
- **Path Optimization**: Track `best_path_positions` set for breadcrumb following
- **Stuck Detection**: Force MOVE_FORWARD after 5 repeated positions near start

## Quality Gates
- **Build**: Ensure Python 3.x compatibility, no syntax errors
- **Tests**: Run `python main.py` - all scenarios should pass without errors
- **Linting**: Follow PEP 8, use type hints, avoid bare `except` clauses
- **Performance**: Keep step processing under 0.001s per agent

## Debugging Tips
- Check `agent.action_history` for decision breakdowns
- Monitor consciousness trajectories in test results
- Use ASCII visualization to verify maze navigation logic
- Examine memory influence calculations for unexpected behavior