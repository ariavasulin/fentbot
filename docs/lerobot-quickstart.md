# LeRobot SO-101 Quickstart Guide

## Physical AI Hackathon - Poker Bot Project

**Date**: January 31, 2026
**Hardware**: LeRobot SO-101 Robotic Arm
**Framework**: LeRobot + Solo CLI

---

## TL;DR - Get Running in 5 Commands

```bash
# 1. Install Solo CLI (easiest path)
uv pip install solo-cli

# 2. Calibrate both arms
solo robo --calibrate both

# 3. Test teleoperation
solo robo --teleop

# 4. Record training data
solo robo --record

# 5. Train a policy
solo robo --train
```

---

## Two Paths: Solo CLI vs LeRobot CLI

| Feature | Solo CLI | LeRobot CLI |
|---------|----------|-------------|
| **Installation** | `uv pip install solo-cli` | `pip install lerobot` |
| **Interface** | Interactive prompts | Command-line flags |
| **Best for** | Quick start, hackathons | Full control, customization |
| **Motors pre-configured?** | Usually yes | Manual setup required |

**Recommendation for hackathon**: Start with Solo CLI for fastest setup.

---

## Step 1: Installation

### Solo CLI (Recommended)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
uv pip install solo-cli

# Verify
solo --help
```

### LeRobot (Alternative)

```bash
# Clone repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create conda environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg (required)
conda install ffmpeg -c conda-forge

# Install with Feetech motor support
pip install -e ".[feetech]"
```

---

## Step 2: Find Your Ports

### With Solo CLI
```bash
solo robo --calibrate both
# Follow interactive prompts - ports auto-detected
```

### With LeRobot
```bash
lerobot-find-port
```

**Linux users**: Grant port access:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

---

## Step 3: Calibrate Arms

### With Solo CLI
```bash
# Calibrate both arms interactively
solo robo --calibrate both

# Or individually
solo robo --calibrate leader
solo robo --calibrate follower
```

### With LeRobot
```bash
# Calibrate follower
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower

# Calibrate leader
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader
```

### Calibration Process
1. Move arm to middle position (all joints at midpoint)
2. Press Enter
3. Move each joint through its full range
4. System records MIN/MAX values

---

## Step 4: Test Teleoperation

```bash
# Solo CLI
solo robo --teleop

# LeRobot
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader
```

Move the leader arm - the follower should mirror your movements.

---

## Step 5: Set Up Cameras

### Find Camera IDs
```bash
lerobot-find-cameras opencv
```

### Configure in Commands
```bash
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

### Multiple Cameras
```bash
--robot.cameras="{
    front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}
}"
```

---

## Step 6: Record Training Data

### With Solo CLI
```bash
solo robo --record
# Follow prompts for:
# - HuggingFace upload (optional)
# - Dataset name
# - Task description
# - Episode duration
# - Number of episodes
```

### With LeRobot
```bash
# Login to HuggingFace first
huggingface-cli login

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/poker_bot_data \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the poker chip and place it in the stack"
```

### Recording Tips
- Record 50+ episodes for good results
- Keep movements consistent
- Ensure camera can see the workspace
- Press Right Arrow to skip to next episode
- Press ESC to finish recording

---

## Step 7: Train a Policy

### With Solo CLI
```bash
solo robo --train
# Select:
# - Dataset (local or HuggingFace Hub)
# - Policy type: ACT, SmolVLA, Diffusion, etc.
# - Training steps
# - Batch size
```

### With LeRobot
```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/poker_bot_data \
    --policy.type=act \
    --output_dir=outputs/train/act_poker \
    --policy.device=cuda
```

### Policy Choices

| Policy | Parameters | Best For |
|--------|------------|----------|
| **ACT** | 80M | Quick training, single tasks |
| **SmolVLA** | 450M | Language-conditioned, runs on CPU |
| **Diffusion** | ~100M | Complex manipulation |

**For hackathon**: Start with ACT (fastest to train).

---

## Step 8: Run Inference

### With Solo CLI
```bash
solo robo --inference
# Provide:
# - Policy path (HuggingFace model ID or local)
# - Task description
# - Duration
```

### With LeRobot
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_poker \
    --dataset.single_task="Pick up the poker chip" \
    --dataset.num_episodes=10 \
    --control.policy.path=${HF_USER}/act_poker_policy
```

---

## Troubleshooting

### "Missing motor IDs" error
```bash
solo robo --motors both
```

### Port permission denied (Linux)
```bash
sudo chmod 666 /dev/ttyACM*
```

### Camera not found
```bash
lerobot-find-cameras opencv
# Check USB connections, try different ports
```

### Training out of memory
- Reduce batch size: `--batch_size=4`
- Use CPU: `--policy.device=cpu`

### Robot not moving smoothly
- Re-calibrate: `solo robo --calibrate both`
- Check for mechanical obstructions

---

## GPU & Cloud Resources (Hackathon)

- **Velda Cloud**: https://physical-ai.velda.cloud/
- **VESSL AI**: https://cloud.vessl.ai/

---

## Key Links

- **Solo Tech Docs**: https://docs.getsolo.tech/welcome
- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **LeRobot Docs**: https://huggingface.co/docs/lerobot
- **SO-101 Guide**: https://huggingface.co/docs/lerobot/so101
- **SO-ARM100 Hardware**: https://github.com/TheRobotStudio/SO-ARM100
