# Chip Manipulation Primitives Implementation Plan

## Overview

Train two ACT policies for SO-101 to perform **vertical-only** chip manipulation:
- **`pick_chip`**: Descend to chip stack, grab top chip, lift to safe height
- **`drop_chip`**: Descend to target, release chip, lift to safe height

These primitives are **position-agnostic** - they only handle vertical motion. Horizontal positioning is handled by a separate hardcoded grid navigation system.

### System Architecture

```
move_chip(from_xy, to_xy):
    1. [HARDCODED] Move arm to from_xy at safe height (grid calibration lookup)
    2. [LEARNED]   pick_chip(): descend → grab → lift (variable stack height)
    3. [HARDCODED] Move arm to to_xy at safe height (grid calibration lookup)
    4. [LEARNED]   drop_chip(): descend → release → lift (variable stack height)
```

**Key insight**: The learned policies only see "I'm above a stack, go down and grab/release." They don't know or care WHERE on the grid they are. Train once at any fixed position, deploy everywhere via hardcoded horizontal movement.

## Out of Scope

- Board state tracking / computer vision for chip detection
- Game logic and decision making
- Strategy or move planning
- Multi-chip manipulation in single action
- Horizontal movement (handled by grid calibration, not learning)

---

## Open Questions

> These must be resolved before or during implementation.

| # | Question | Options/Notes |
|---|----------|---------------|
| 1 | Which edge is robot mounted on? | Short (24") or long (36") side |
| 2 | What is the usable working zone? | Limited by ~12-14" SO-101 reach. Suggest 8x8 or 10x10 grid subset |
| 3 | Chip thickness? | Needed for stack height calculations (standard poker chip ~3.3mm) |
| 4 | Camera count and positions? | Recommend: 1-2 cameras (front + top or wrist) |
| 5 | Gripper opening for chip? | Need to measure chip diameter (~39mm for standard) |
| 6 | Training position? | Which grid position to use for demonstration recording? |

---

## Hardware Setup

### Physical Layout

```
+------------------+
|      [Pot]       |    <- Example: pot at some grid position
|   23 x 35 grid   |    <- 24" x 36" paper, 1/2" borders, 1" spacing
|   (1" squares)   |
|  [CHIP SOURCE]   |    <- Example: player's chips at another position
+------------------+
        ^
        |
    [SO-101 ARM]   <- Mounted at edge
```

Both pot and chip source are just grid positions - the system can move chips between ANY two grid positions.

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
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.10
uv venv --python 3.10
source .venv/bin/activate

# Install LeRobot with Feetech support
uv pip install 'lerobot[feetech]'

# Install FFmpeg (required for video encoding)
# macOS:
brew install ffmpeg
# Linux:
sudo apt-get install ffmpeg

# Verify installation
python -c "import lerobot; print(lerobot.__version__)"
```

**Alternative: Install from source (recommended for development)**
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e ".[feetech]"
```

### 1.2 Hardware Connection

```bash
# Find USB ports (cross-platform)
lerobot-find-port

# Or manually check:
# Linux: ls /dev/ttyACM* /dev/ttyUSB*
# macOS: ls /dev/tty.usbmodem*
# Windows: Check Device Manager for COM ports

# Expected: two ports (leader + follower)
# e.g., /dev/ttyACM0 (follower), /dev/ttyACM1 (leader)

# Linux: Grant permissions
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
# Or add user to dialout group permanently:
sudo usermod -a -G dialout $USER
```

### 1.3 Motor Setup (First Time Only)

Set motor IDs and baudrates. Connect ONE motor at a time:

```bash
# Setup follower arm motors
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# Setup leader arm motors
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
```

### 1.4 Arm Calibration

Run LeRobot's built-in calibration for both arms:

```bash
# Calibrate follower arm
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower

# Calibrate leader arm
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader
```

Calibration files saved to: `~/.cache/huggingface/lerobot/calibration/`

### 1.5 Test Teleoperation

Verify leader-follower mirroring works:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader
```

### Success Criteria

- [ ] LeRobot installed and importable (`python -c "import lerobot"`)
- [ ] Both arms detected via `lerobot-find-port`
- [ ] Motor setup complete (IDs and baudrates configured)
- [ ] Calibration files saved to `~/.cache/huggingface/lerobot/calibration/`
- [ ] Teleoperation works: leader arm mirrors to follower

---

## Phase 2: Grid Calibration System

### 2.1 Purpose

Create a mapping from grid coordinates (row, col) to robot joint positions at safe height (10"). This enables **hardcoded horizontal navigation** between any two grid positions.

**Note**: This is NOT used as policy input. Policies are position-agnostic.

### 2.2 Approach

**4-corner manual calibration + bilinear interpolation**

```
Corner positions to record (at safe height):
(0,0) -------- (0, max_col)
  |                |
  |    WORKING     |
  |     ZONE       |
  |                |
(max_row, 0) -- (max_row, max_col)
```

### 2.3 Calibration Script Specification

**File**: `scripts/calibrate_grid.py`

**Inputs**:
- Working zone dimensions (rows, cols) - default 8x8
- Safe height above grid (default: 10 inches / 254mm)

**Process**:
1. Prompt user to move arm (via teleoperation) to corner (0,0) at safe height
2. Press key to record joint positions
3. Repeat for remaining 3 corners
4. Compute interpolated positions for all grid cells using bilinear interpolation
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
    "0,0": [j1, j2, j3, j4, j5, j6],
    "0,1": [...],
    ...
  }
}
```

### 2.4 Grid Navigation Function

**File**: `scripts/grid_nav.py`

```python
def move_to_grid_position(robot, calibration, row, col):
    """Move arm to (row, col) at safe height using calibrated joint positions."""
    target_joints = calibration["positions"][f"{row},{col}"]
    robot.move_to(target_joints)
```

This is simple joint-space interpolation - no learning required.

### 2.5 Verification Script

**File**: `scripts/verify_grid.py`

Move arm sequentially to a sample of grid positions (e.g., corners + center + random) to visually verify accuracy.

### Success Criteria

- [ ] Calibration script records 4 corners
- [ ] Interpolated positions computed for full grid
- [ ] Verification shows arm reaches all positions accurately (within ~5mm)
- [ ] Calibration file saved and loadable
- [ ] `move_to_grid_position()` function works

---

## Phase 3: Data Collection

### 3.1 Key Insight: Position-Agnostic Training

Since policies only handle **vertical motion**, we train at a **single fixed grid position**. The policy learns:
- How to descend based on visual observation of stack height
- When to close/open gripper
- How to lift back to safe height

This generalizes to all grid positions because vertical motion is the same everywhere.

### 3.2 Dataset Structure

Two separate datasets:

```
data/
├── pick_chip/
│   ├── episode_0/
│   │   ├── observation.images.front/
│   │   ├── observation.state/
│   │   ├── action/
│   │   └── metadata.json  # {stack_height: N}
│   ├── episode_1/
│   └── ...
└── drop_chip/
    ├── episode_0/
    └── ...
```

**Note**: No grid position in metadata - policies are position-agnostic.

### 3.3 Recording with LeRobot CLI

**Basic recording command:**

```bash
# Record pick_chip demonstrations
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --dataset.repo_id=${HF_USER}/pick_chip \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up top chip from stack" \
    --display_data=true

# Record drop_chip demonstrations
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --dataset.repo_id=${HF_USER}/drop_chip \
    --dataset.num_episodes=50 \
    --dataset.single_task="Drop chip onto stack" \
    --display_data=true
```

**Find available cameras:**
```bash
lerobot-find-cameras opencv
```

### 3.4 Recording Process

**Process for `pick_chip`**:
1. Position arm at training location at safe height (manually before starting)
2. Set up stack of N chips (vary height 1-10 across episodes)
3. Start recording (press Enter)
4. Teleoperate: descend → grab → lift to safe height
5. Stop recording (press Enter or 's')
6. Repeat with different stack heights

**Process for `drop_chip`**:
1. Start with chip in gripper at safe height above target
2. Set up existing stack of N chips at target (vary 0-9 across episodes)
3. Start recording
4. Teleoperate: descend → release → lift to safe height
5. Stop recording
6. Repeat

### 3.5 Training Variables

| Variable | Range | Purpose |
|----------|-------|---------|
| Stack height (pick) | 1-10 chips | Policy learns to descend appropriate amount |
| Stack height (drop) | 0-9 chips | Policy learns to descend to existing stack |
| Natural jitter | N/A | Human demonstrations naturally vary |

### 3.6 Demonstration Count Targets

| Primitive | Target Episodes | Notes |
|-----------|-----------------|-------|
| `pick_chip` | 50-100 | Uniform distribution across stack heights 1-10 |
| `drop_chip` | 50-100 | Uniform distribution across stack heights 0-9 |

~5-10 demos per stack height should be sufficient.

### Success Criteria

- [ ] Recording script functional
- [ ] Episodes include stack height metadata
- [ ] At least 50 episodes per primitive
- [ ] Uniform distribution across stack heights
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

observation:
  images: [front_camera]  # Camera input(s) - TBD based on setup
  state: [joint_positions, gripper_position]
  # NO position conditioning - policies are position-agnostic
```

### 4.2 Training Commands

```bash
# Train pick_chip policy
lerobot-train \
    --dataset.repo_id=${HF_USER}/pick_chip \
    --policy.type=act \
    --output_dir=outputs/train/pick_chip \
    --job_name=pick_chip \
    --policy.device=cuda

# Train drop_chip policy
lerobot-train \
    --dataset.repo_id=${HF_USER}/drop_chip \
    --policy.type=act \
    --output_dir=outputs/train/drop_chip \
    --job_name=drop_chip \
    --policy.device=cuda
```

**For Apple Silicon (M1/M2/M3):**
```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/pick_chip \
    --policy.type=act \
    --output_dir=outputs/train/pick_chip \
    --policy.device=mps
```

### 4.3 Training Monitoring

Use Weights & Biases (optional but recommended):

```bash
wandb login

# Add wandb flag to training command
lerobot-train \
    --dataset.repo_id=${HF_USER}/pick_chip \
    --policy.type=act \
    --output_dir=outputs/train/pick_chip \
    --wandb.enable=true
```

Monitor for:
- Loss convergence
- Action prediction accuracy
- Overfitting (train vs. validation loss)

### 4.4 Checkpoints and Resuming

Checkpoints saved to: `outputs/train/<job_name>/checkpoints/`

```bash
# Resume from checkpoint
lerobot-train \
    --config_path=outputs/train/pick_chip/checkpoints/last/pretrained_model/train_config.json \
    --resume=true

# Upload trained model to Hub
huggingface-cli upload ${HF_USER}/pick_chip_policy \
    outputs/train/pick_chip/checkpoints/last/pretrained_model
```

### Success Criteria

- [ ] Training completes without errors
- [ ] Loss converges (final loss < initial loss by 10x+)
- [ ] Model checkpoints saved to `outputs/train/`
- [ ] Models uploaded to HuggingFace Hub

---

## Phase 5: Integration & Deployment

### 5.1 The `move_chip` Function

**File**: `scripts/move_chip.py`

This is the main orchestration function that combines hardcoded navigation with learned primitives:

```python
def move_chip(robot, grid_calibration, pick_policy, drop_policy, from_pos, to_pos):
    """
    Move a chip from from_pos to to_pos.

    Args:
        robot: LeRobot robot instance
        grid_calibration: Loaded calibration data
        pick_policy: Trained pick_chip ACT policy
        drop_policy: Trained drop_chip ACT policy
        from_pos: (row, col) tuple - source position
        to_pos: (row, col) tuple - destination position
    """
    # Step 1: Move to source position (HARDCODED)
    move_to_grid_position(robot, grid_calibration, from_pos[0], from_pos[1])

    # Step 2: Pick chip (LEARNED)
    run_policy(robot, pick_policy, termination="gripper_closed_and_lifted")

    # Step 3: Move to destination position (HARDCODED)
    move_to_grid_position(robot, grid_calibration, to_pos[0], to_pos[1])

    # Step 4: Drop chip (LEARNED)
    run_policy(robot, drop_policy, termination="gripper_open_and_lifted")
```

### 5.2 Policy Inference

**File**: `scripts/run_policy.py`

```python
def run_policy(robot, policy, termination):
    """
    Run a learned policy until termination condition.

    Termination conditions:
    - "gripper_closed_and_lifted": Gripper closed AND z at safe height
    - "gripper_open_and_lifted": Gripper open AND z at safe height
    - Timeout (5 seconds fallback)
    """
    while not check_termination(robot, termination):
        observation = get_observation(robot)  # camera + joint state
        action = policy.predict(observation)
        robot.execute(action)
```

### 5.3 Termination Conditions

| Primitive | Condition |
|-----------|-----------|
| `pick_chip` | Gripper closed AND z-position at safe height |
| `drop_chip` | Gripper open AND z-position at safe height |
| Both | Timeout after 5 seconds (fallback) |

### 5.4 Evaluation Protocol

Test the full `move_chip` function:

1. **Same position as training**: 10 trials moving chips between training position and another
2. **Different positions**: 10 trials at positions NOT used during training
3. **Various stack heights**: Test with 2, 5, and 9 chip stacks

**Metrics**:
- Success rate (chip successfully moved)
- Completion time
- Any dropped chips or misses

### Success Criteria

- [ ] `move_chip()` function works end-to-end
- [ ] Pick success rate > 80%
- [ ] Drop success rate > 80%
- [ ] Policies generalize to unseen grid positions (same success rate)

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
│   ├── calibrate_grid.py          # Phase 2: Grid calibration
│   ├── grid_nav.py                # Phase 2: Hardcoded navigation
│   ├── verify_grid.py             # Phase 2: Verification
│   ├── record_demonstrations.py   # Phase 3: Data collection
│   ├── run_policy.py              # Phase 5: Policy inference
│   └── move_chip.py               # Phase 5: Main orchestration
├── configs/
│   ├── pick_chip_act.yaml         # Phase 4: Training config
│   └── drop_chip_act.yaml         # Phase 4: Training config
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
2. **Phase 2**: Grid calibration system (calibrate, verify, navigate)
3. **Phase 3**: Data collection (50+ demos per primitive at ONE position)
4. **Phase 4**: Policy training (ACT, ~2 hours each)
5. **Phase 5**: Integration and deployment (move_chip function)

---

## Summary: What's Learned vs. Hardcoded

| Component | Type | Description |
|-----------|------|-------------|
| Horizontal navigation | **Hardcoded** | Move arm to any (x,y) via grid calibration |
| `pick_chip` | **Learned** | Vertical: descend, grab, lift |
| `drop_chip` | **Learned** | Vertical: descend, release, lift |
| `move_chip` | **Orchestration** | Combines hardcoded + learned |

The policies are trained once at a single position and generalize everywhere because they only learn vertical motion, which is position-independent.

---

## References

- LeRobot documentation: https://github.com/huggingface/lerobot
- ACT paper: https://arxiv.org/abs/2304.13705
- SO-101 specs: See `docs/REFERENCE.md`
