# Learning-Based Navigation Improvements

## Changes Made to Enable Emergent Behavior

Instead of hardcoding "don't walk into walls" or "follow line of sight", the agents now learn these behaviors through **stronger reward/penalty signals**.

### 1. **Stronger Learning Signals**

**Collisions** (hitting walls):
- Reward: `-0.5` (was `-0.1`) → 5x stronger penalty
- Surprise: `0.8` (was `0.3`) → More memorable
- Emotional Impact: `-0.8` → Strong negative feeling
- Result: Agents quickly learn "moving forward at position X facing direction Y = BAD"

**Successful Moves**:
- Reward: `+0.3` for new cells, `+0.15` for revisiting (was `+0.1/+0.05`)
- Emotional Impact: `+0.6` → Positive reinforcement
- Result: Agents learn "this direction from this position = GOOD"

**Goal Achievement**:
- Reward: `+10.0` (unchanged, but now stands out more)
- Emotional Impact: `+1.0` → Maximum positive
- Result: Path to goal becomes strongly reinforced

### 2. **Turn Cost Increased**

- Turning reward: `-0.05` (was `-0.005`) → 10x stronger penalty
- Result: Agents avoid spinning endlessly, prefer forward progress

### 3. **Equal Base Weights**

All actions start at `base_weight = 1.0`:
- Move forward: 1.0
- Turn left: 1.0  
- Turn right: 1.0

**The 13-factor decision system does all the work**:
- Factor 5 (Memory): "I hit a wall last time I tried this" → 0.8x penalty
- Factor 5b (Collective): "3 other agents hit walls here" → stronger avoidance
- Factor 1 (Goal): "This direction is toward the exit" → +10 boost
- Factor 10 (Risk): "High reward/penalty makes this risky" → careful evaluation

### 4. **Real-Time Broadcasting**

When Agent A hits a wall:
1. Creates memory: `collision at pos_(5,5), dir_0` with -0.5 reward
2. Broadcasts immediately to collective pool
3. Agent B approaches same position facing same direction
4. Queries collective: "What happened at pos_(5,5) with dir_0?"
5. Finds collision memory with -0.8 emotional impact
6. Memory influence applies 0.8x penalty to "move forward"
7. Turns instead!

### 5. **Learning Accumulation**

**First collision** at a position:
- Personal memory: -0.8 emotional impact
- Memory influence: ~0.85x multiplier (mild avoidance)

**After 3 agents hit same wall**:
- Collective memory: 3 contributors, 0.0 reliability (0 successes / 3 attempts)
- Collective influence: ~0.80x multiplier
- Combined with personal: ~0.68x total
- Strong avoidance!

**After someone succeeds at a position**:
- Personal memory: +0.6 emotional impact
- Reliability: increases toward 1.0
- Memory influence: ~1.3x multiplier (favor this action)
- Agents prefer successful paths

### 6. **Why They Should Learn**

**Problem**: "Why do they spin and go backwards?"

**Root Cause**: 
- Old penalties were too weak (-0.1) to form strong memories
- Base weights were too prescriptive (hardcoded obstacle avoidance)
- Turn cost was negligible (-0.005) so spinning was "free"

**Solution**:
- Collisions now memorable (-0.5 reward, 0.8 surprise, -0.8 emotion)
- Success is rewarding (+0.3 reward, +0.6 emotion)
- Turning costs something (-0.05) so they think twice
- Base weights are neutral - learning shapes behavior
- Collective memory amplifies learning across group

**Expected Behavior**:
1. Early agents explore randomly, hit many walls
2. Each collision broadcasts "don't go this way!"
3. Later agents avoid known dead ends
4. First success broadcasts "this path works!"
5. Subsequent agents follow successful paths
6. Success rate climbs from 33% → 60%+ over generations

### 7. **Test It**

```bash
python test_collective_learning.py --agents 10 --respawns 3
```

Watch for:
- Early generations: Lots of collisions, random exploration
- Mid generations: Fewer repeat mistakes, some successes
- Late generations: Most agents follow learned paths, high success rate
- Collective memory growing: Should see 100+ shared experiences

If they still spin too much:
- Increase turn penalty further (try -0.1 or -0.2)
- Lower broadcast threshold (share more experiences)
- Increase collision penalty (try -0.8 or -1.0)
