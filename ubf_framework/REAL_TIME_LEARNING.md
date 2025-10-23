# Real-Time Collective Learning System

## Overview
Agents now share knowledge **continuously as they explore**, not just when reaching the exit. This creates a dynamic group intelligence where discoveries propagate instantly.

## How It Works

### 1. **Real-Time Broadcasting**
Every time an agent has a significant experience (significance ≥ 0.12), they immediately broadcast it to the collective memory pool:

- **Collision with wall**: "Don't try moving forward at position (3,5) facing North!"
- **Successful move**: "Moving forward at (2,4) facing East worked!"
- **New discovery**: "Found unexplored area at (7,8)!"

### 2. **What Gets Shared**
```
Position: Exact (x,y) coordinates
Direction: Which way the agent was facing (N/E/S/W)
Action: What they tried to do
Outcome: Success, collision, or discovery
Significance: How important this experience was
```

### 3. **Anti-Spam Protection**
- Agents won't broadcast the same position multiple times (unless critical)
- Only significant experiences (≥ 0.12 threshold) get shared
- Collisions and goal achievements always broadcast

### 4. **Learning Flow**

```
Agent A hits wall at (5,5) facing North
    ↓
Immediately broadcasts: "collision at (5,5), dir_0, action_move_forward"
    ↓
Agent B is at (5,4) considering moving North
    ↓
Queries collective memory: "Any experiences at pos_(5,5) with dir_0?"
    ↓
Finds Agent A's collision memory (negative emotional impact)
    ↓
Applies 0.8x penalty to "move forward" action
    ↓
Agent B turns instead! Avoids the wall without hitting it!
```

### 5. **Continuous Growth**
The collective memory pool grows **every step** as agents explore:
- Generation 1: ~20-30 memories (pioneers discovering)
- Generation 2: ~50-80 memories (building on discoveries)
- Generation 3: ~100+ memories (comprehensive knowledge)

### 6. **Success Amplification**
When an agent **does** reach the exit, they broadcast their top 20 most significant memories as a "complete path package" - but by then, most of their knowledge has already been shared!

## Benefits

✅ **Faster Learning**: No need to wait for someone to succeed before sharing
✅ **Avoid Redundant Mistakes**: Agent B doesn't hit the same wall Agent A hit
✅ **Collective Exploration**: Group coverage increases exponentially
✅ **Emergency Knowledge**: Even failed runs contribute valuable "what NOT to do" data
✅ **Distributed Intelligence**: The group becomes smarter than any individual

## Example Output
```
Generation 1 - 10 active agents
  [Step 10] +5 shared experiences (total: 15)
  [Step 20] +8 shared experiences (total: 23)
  [Step 30] +3 shared experiences (total: 26)
  ✓ agent_2 REACHED EXIT! (steps: 42, respawns: 0)
  
Generation 1 complete:
  Successes: 1
  Real-Time Broadcasts: 38
  Collective Memories: 64 (grew by 64)
```

## Key Parameters

- `broadcast_threshold = 0.12` - Minimum significance to share (lower = more sharing)
- `collective_factor = 0.7x` - Collective memories weighted at 70% of personal memories
- `min_reliability = 0.3` - Only query collective memories with ≥30% success rate

## Run It
```bash
python test_collective_learning.py --agents 10 --respawns 3 --steps 100
```

Watch as the collective memory pool grows in real-time and success rates climb across generations!
