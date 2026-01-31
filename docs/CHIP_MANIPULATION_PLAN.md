# Chip Manipulation Primitives Implementation Plan

## Overview

Train two position-conditioned ACT policies for SO-101 to perform poker chip manipulation:
- **`pick_chip`**: Descend to chip stack, grab top chip, lift to safe height
- **`drop_chip`**: Descend to target position, release chip, lift to safe height

These are low-level manipulation primitives. A higher-level system (out of scope) will call these with grid coordinates.

## Out of Scope

- Board state tracking / computer vision for chip detection
- Game logic and decision making
- Strategy or move planning
- Multi-chip manipulation in single action

---

## Open Questions

> These must be resolved before or during implementation.

| # | Question | Options/Notes |
|---|----------|---------------|
| 1 | Which edge is robot mounted on? | Short (24") or long (36") side |
| 2 | What is the usable working zone? | Limited by ~12-14" SO-101 reach. Suggest 8x8 or 10x10 grid subset |
| 3 | Chip thickness? | Needed for stack height calculations (standard poker chip ~3.3mm) |
| 4 | Chip source location? | Where relative to grid? Adjacent corner? |
| 5 | Camera count and positions? | Recommend: 1-2 cameras (front + top or wrist) |
| 6 | Gripper opening for chip? | Need to measure chip diameter (~39mm for standard) |

---

## Hardware Setup

### Physical Layout

```
+------------------+
|                  |
|   23 x 35 grid   |    <- 24" x 36" paper, 1/2" borders, 1" spacing
|   (1" squares)   |
|                  |
+------------------+
        ^
        |
    [SO-101 ARM]   <- Mounted at edge
        |
   [CHIP SOURCE]   <- Dedicated pickup location
```

### Equipment Required

- SO-101 dual-arm system (leader + follower)
- USB connections to both arms
- 1-2 USB cameras (TBD positions)
- Grid paper (24" x 36")
- Poker chips (quantity: 20+)
- Rigid mounting for arm at grid edge

---

## Phase 1: Environment Setup

### 1.1 LeRobot Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install LeRobot with Feetech support
pip install lerobot[feetech]

# Verify installation
python -c "import lerobot; print(lerobot.__version__)"
```

### 1.2 Hardware Connection

```bash
# Find USB ports
ls /dev/ttyACM* /dev/ttyUSB*

# Expected: two ports (leader + follower)
# e.g., /dev/ttyACM0 (leader), /dev/ttyACM1 (follower)
```

### 1.3 Arm Calibration

Run LeRobot's built-in calibration for both arms:

```bash
# Calibrate follower arm
python -m lerobot.scripts.control_robot calibrate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' --arms follower

# Calibrate leader arm
python -m lerobot.scripts.control_robot calibrate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' --arms leader
```

### Success Criteria

- [ ] LeRobot installed and importable
- [ ] Both arms detected on USB ports
- [ ] Calibration files saved to `~/.cache/huggingface/lerobot/calibration/`
- [ ] Teleoperation works: leader arm mirrors to follower

---

## Phase 2: Grid Calibration System

### 2.1 Concept

Create a mapping from grid coordinates (row, col) to robot joint positions at safe height (10").

**Approach**: 4-corner manual calibration + bilinear interpolation

```
Corner positions to record:
(0,0) -------- (0, max_col)
  |                |
  |    GRID        |
  |                |
(max_row, 0) -- (max_row, max_col)
```

### 2.2 Calibration Script Specification

**File**: `scripts/calibrate_grid.py`

**Inputs**:
- Working zone dimensions (rows, cols) - default 8x8
- Safe height above grid (default: 10 inches / 254mm)

**Process**:
1. Prompt user to move arm (via teleoperation) to corner (0,0) at safe height
2. Press key to record joint positions
3. Repeat for remaining 3 corners
4. Compute interpolated positions for all grid cells
5. Save calibration file

**Output**: `config/grid_calibration.json`

```json
{
  "metadata": {
    "rows": 8,
    "cols": 8,
    "safe_height_mm": 254,
    "calibrated_at": "2026-01-31T...",
    "corners": {
      "top_left": [j1, j2, j3, j4, j5, j6],
      "top_right": [...],
      "bottom_left": [...],
      "bottom_right": [...]
    }
  },
  "positions": {
    "0,0": {"safe": [j1, j2, j3, j4, j5, j6]},
    "0,1": {"safe": [...]},
    ...
  }
}
```

### 2.3 Chip Source Calibration

**Additional calibration point**: The chip source location (off-grid).

Add to calibration script:
1. Move arm to chip source at safe height → record
2. Move arm to chip source at pickup height (stack of 10) → record
3. Save as special position in calibration file

### 2.4 Verification Script

**File**: `scripts/verify_grid.py`

Move arm to each calibrated position sequentially to visually verify accuracy.

### Success Criteria

- [ ] Calibration script records 4 corners + chip source
- [ ] Interpolated positions computed for full grid
- [ ] Verification shows arm reaches all positions accurately (within ~5mm)
- [ ] Calibration file saved and loadable

---

## Phase 3: Data Collection

### 3.1 Dataset Structure

Two separate datasets:

```
data/
├── pick_chip/
│   ├── episode_0/
│   │   ├── observation.images.front/  # or however LeRobot structures it
│   │   ├── observation.state/
│   │   ├── action/
│   │   └── metadata.json  # {grid_pos: [r,c], stack_height: N}
│   ├── episode_1/
│   └── ...
└── drop_chip/
    ├── episode_0/
    └── ...
```

### 3.2 Recording Script Specification

**File**: `scripts/record_demonstrations.py`

**Arguments**:
- `--primitive`: `pick_chip` or `drop_chip`
- `--num-episodes`: Target episode count
- `--grid-calibration`: Path to calibration file

**Process for `pick_chip`**:
1. System selects random grid position and stack height (1-10)
2. Display target to operator: "Pick chip from (3, 5), stack height: 4"
3. Human manually sets up chips at that position with correct height
4. System moves arm to chip source (or starting position)
5. Start recording
6. Operator teleoperates: descend → grab → lift to safe height
7. Stop recording
8. Save episode with metadata
9. Repeat

**Process for `drop_chip`**:
1. System selects random grid position
2. Display target: "Drop chip at (2, 7)"
3. Operator starts with chip in gripper at safe height
4. Start recording
5. Operator teleoperates: move to position → descend → release → lift
6. Stop recording
7. Save episode with metadata
8. Repeat

### 3.3 Data Augmentation Considerations

Position conditioning requires diverse training positions. Ensure:
- Minimum 10-20 unique grid positions sampled
- Uniform distribution across the working zone
- Stack heights uniformly distributed 1-10

### 3.4 Demonstration Count Targets

| Primitive | Target Episodes | Notes |
|-----------|-----------------|-------|
| `pick_chip` | 50-100 | Vary position (10+) and height (1-10) |
| `drop_chip` | 50-100 | Vary target position (10+) |

### Success Criteria

- [ ] Recording script functional
- [ ] Episodes include correct metadata (grid position, stack height)
- [ ] At least 50 episodes per primitive
- [ ] Diverse positions and stack heights represented
- [ ] Data format compatible with LeRobot training

---

## Phase 4: Policy Training

### 4.1 ACT Policy Configuration

ACT (Action Chunking with Transformers) is recommended for:
- Fast training (~2 hours on consumer GPU)
- Good performance on precise manipulation
- Native LeRobot support

**Key hyperparameters**:

```yaml
policy:
  name: act
  chunk_size: 100        # Action sequence length
  n_obs_steps: 1         # Observation history
  dim_model: 512         # Transformer dimension
  n_heads: 8             # Attention heads
  n_layers: 4            # Transformer layers

training:
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 2000

# Position conditioning - add to observation space
observation:
  images: [front_camera]  # Camera input(s)
  state: [joint_positions, gripper_position]
  context: [target_row_normalized, target_col_normalized]  # 0-1 floats
```

### 4.2 Position Conditioning Implementation

Modify observation to include normalized grid coordinates:

```python
# During data collection, add to each timestep:
observation["target_position"] = [
    row / max_rows,  # Normalized to [0, 1]
    col / max_cols
]
```

This allows the policy to generalize: same weights, different target positions.

### 4.3 Training Commands

```bash
# Train pick_chip policy
python -m lerobot.scripts.train \
    --dataset-repo-id local:data/pick_chip \
    --policy-name act \
    --output-dir outputs/pick_chip \
    --config-path configs/pick_chip_act.yaml

# Train drop_chip policy
python -m lerobot.scripts.train \
    --dataset-repo-id local:data/drop_chip \
    --policy-name act \
    --output-dir outputs/drop_chip \
    --config-path configs/drop_chip_act.yaml
```

### 4.4 Training Monitoring

Use Weights & Biases (optional but recommended):

```bash
wandb login
# Add --wandb flag to training command
```

Monitor for:
- Loss convergence
- Action prediction accuracy
- Overfitting (train vs. validation loss)

### Success Criteria

- [ ] Training configs created for both primitives
- [ ] Position conditioning integrated into observation space
- [ ] Training completes without errors
- [ ] Loss converges (final loss < initial loss by 10x+)
- [ ] Model checkpoints saved

---

## Phase 5: Policy Evaluation & Deployment

### 5.1 Inference Script Specification

**File**: `scripts/run_policy.py`

**Arguments**:
- `--primitive`: `pick_chip` or `drop_chip`
- `--checkpoint`: Path to trained model
- `--grid-position`: Target position as "row,col"
- `--stack-height`: (for pick_chip) Expected stack height

**Process**:
1. Load policy checkpoint
2. Load grid calibration
3. Set target position in observation
4. Run inference loop:
   - Get camera observation
   - Get robot state
   - Add target position to observation
   - Policy predicts action chunk
   - Execute actions on robot
   - Repeat until termination condition

### 5.2 Termination Conditions

**pick_chip**:
- Gripper closed AND z-position at safe height
- OR timeout (5 seconds)

**drop_chip**:
- Gripper open AND z-position at safe height
- OR timeout (5 seconds)

### 5.3 Evaluation Protocol

Test each policy on:
- 5 seen positions (from training data)
- 5 unseen positions (interpolated, not in training)
- 3 stack heights (low: 2, medium: 5, high: 9)

**Metrics**:
- Success rate (chip picked/placed correctly)
- Position accuracy (distance from target)
- Completion time

### Success Criteria

- [ ] Inference script runs trained policies
- [ ] Pick success rate > 80% on seen positions
- [ ] Drop success rate > 80% on seen positions
- [ ] Generalization to unseen positions works (> 60% success)

---

## File Structure (Final)

```
fentbot/
├── docs/
│   ├── QUICKSTART.md
│   ├── REFERENCE.md
│   ├── VLA_MODELS.md
│   └── CHIP_MANIPULATION_PLAN.md  # This document
├── scripts/
│   ├── calibrate_grid.py          # Phase 2
│   ├── verify_grid.py             # Phase 2
│   ├── record_demonstrations.py   # Phase 3
│   └── run_policy.py              # Phase 5
├── configs/
│   ├── pick_chip_act.yaml         # Phase 4
│   └── drop_chip_act.yaml         # Phase 4
├── config/
│   └── grid_calibration.json      # Generated by calibration
├── data/
│   ├── pick_chip/                 # Recorded demonstrations
│   └── drop_chip/
└── outputs/
    ├── pick_chip/                 # Trained models
    └── drop_chip/
```

---

## Implementation Order

1. **Phase 1**: Environment setup (LeRobot install, arm calibration)
2. **Phase 2**: Grid calibration system
3. **Phase 3**: Data collection (50+ demos per primitive)
4. **Phase 4**: Policy training
5. **Phase 5**: Evaluation and deployment

**Estimated time** (hackathon pace):
- Phase 1: 30 min
- Phase 2: 1-2 hours
- Phase 3: 2-4 hours (depends on demo count)
- Phase 4: 1-2 hours (training time)
- Phase 5: 1 hour

---

## References

- LeRobot documentation: https://github.com/huggingface/lerobot
- ACT paper: https://arxiv.org/abs/2304.13705
- SO-101 specs: See `docs/REFERENCE.md`
