# UBF Maze Navigation System

Universal Behavioral Framework (UBF) implementation for autonomous maze navigation with collective memory and consciousness-based decision making.

## Features

- **Consciousness System**: Agents have coherence, frequency, and temperature-based awareness
- **Collective Memory**: Shared spatial memories with emotional significance
- **Failure Resilience**: Adaptive recovery modes when coherence drops
- **Direct Path Tracking**: Wall positions and successful move tracking
- **Smart Respawn**: Memory-based orientation selection on respawn
- **Breadcrumb Trails**: Agents track and follow their best progress

## Quick Start

```bash
python visualize_ascii.py
```

## Project Structure

- `ubf_framework/core/` - Core UBF components (agent, consciousness, memory)
- `ubf_framework/simulation/` - Maze environment and visualization
- `ubf_framework/models/` - Data models and enums

## Key Components

### Agent System
- 13-factor decision making with memory influence
- Direct wall/path tracking with set-based successful moves
- Breadcrumb trail following for path optimization

### Memory System
- Spatial memory with position and directional awareness
- Memory influence range: 0.3x-2.5x (avoidance to strong preference)
- Significance-based memory filtering

### Consciousness Updates
- Failure resilience: -0.2 frequency penalty, +0.05 coherence recovery
- Recovery mode: Activates at coherence < 0.4 with high exploration noise
- Noise scheduling with 5-step recovery bursts

## Configuration

Agents spawn with identical parameters:
- Frequency: 8.0
- Coherence: 0.75
- Temperature: 1.2
- Respawn trigger: 5 collision failures

## License

MIT
