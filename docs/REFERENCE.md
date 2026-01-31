# LeRobot SO-101 Complete Reference Guide

## Physical AI Hackathon - January 31, 2026

---

## Table of Contents
1. [Hardware Overview](#1-hardware-overview)
2. [Software Stack](#2-software-stack)
3. [Installation](#3-installation)
4. [Calibration](#4-calibration)
5. [Teleoperation](#5-teleoperation)
6. [Recording Datasets](#6-recording-datasets)
7. [Training Policies](#7-training-policies)
8. [Inference](#8-inference)
9. [Camera Configuration](#9-camera-configuration)
10. [VLA Models](#10-vla-models)
11. [Dataset Format](#11-dataset-format)
12. [Troubleshooting](#12-troubleshooting)
13. [Resources](#13-resources)

---

## 1. Hardware Overview

### SO-101 Robot Arm Specifications

| Specification | Value |
|--------------|-------|
| **Degrees of Freedom** | 6 (5 DOF arm + 1 DOF gripper) |
| **Motor Type** | Feetech STS3215 servos |
| **Max Torque** | 30 kg.cm at 12V |
| **Position Resolution** | 12-bit magnetic encoder (4096 steps) |
| **Communication** | Half-duplex TTL serial via USB |
| **Cost** | ~$130-$230 USD |

### Joint Configuration

| Joint # | Name | Leader Gear Ratio | Follower Gear Ratio |
|---------|------|-------------------|---------------------|
| 1 | Base/Shoulder Pan | 1/191 | 1/345 |
| 2 | Shoulder Lift | 1/345 | 1/345 |
| 3 | Elbow Flex | 1/191 | 1/345 |
| 4 | Wrist Flex | 1/147 | 1/345 |
| 5 | Wrist Roll | 1/147 | 1/345 |
| 6 | Gripper | 1/147 | 1/345 |

### Two-Arm System
- **Leader Arm**: Human-operated, used for teleoperation input
- **Follower Arm**: Mirrors leader movements, performs tasks

### Hardware Components

| Component | Quantity | Purpose |
|-----------|----------|---------|
| STS3215 Servos | 12 | 6 per arm |
| Waveshare Control Board | 2 | Motor communication |
| USB-C Cables | 2 | Computer connection |
| 5V Power Supply | 2 | For 7.4V motors |

---

## 2. Software Stack

### Solo CLI
Interactive CLI for robotics operations. Covers motor setup, calibration, teleoperation, recording, training, and inference.

**Commands:**
| Command | Description |
|---------|-------------|
| `solo setup` | Environment configuration |
| `solo robo --motors` | Motor initialization |
| `solo robo --calibrate` | Arm calibration |
| `solo robo --teleop` | Teleoperation mode |
| `solo robo --record` | Dataset recording |
| `solo robo --train` | Policy training |
| `solo robo --inference` | Run trained policy |
| `solo robo --replay` | Replay recorded episodes |

### LeRobot
HuggingFace's open-source robotics framework with state-of-the-art ML tools.

**Commands:**
| Command | Description |
|---------|-------------|
| `lerobot-find-port` | Discover USB ports |
| `lerobot-find-cameras` | Discover cameras |
| `lerobot-setup-motors` | Configure motor IDs |
| `lerobot-calibrate` | Calibrate arm |
| `lerobot-teleoperate` | Teleoperation |
| `lerobot-record` | Record dataset |
| `lerobot-train` | Train policy |
| `lerobot-eval` | Evaluate in simulation |

---

## 3. Installation

### Solo CLI Installation

```bash
# Prerequisites: Python 3.12, uv package manager
uv pip install solo-cli

# Or from source
git clone https://github.com/GetSoloTech/solo-cli.git
cd solo-cli
uv pip install -e .
```

### LeRobot Installation

```bash
# Clone repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg (required for video encoding)
conda install ffmpeg -c conda-forge

# Install with Feetech motor support
pip install -e ".[feetech]"

# For VLA models (Pi0, SmolVLA)
pip install -e ".[pi]"
```

### System Requirements

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 22.04 or macOS |
| Python | 3.10+ |
| CUDA (optional) | 11.8+ for GPU training |

---

## 4. Calibration

### Purpose
Ensures leader and follower arms have matching position values when in the same physical position. Critical for imitation learning.

### Solo CLI Calibration
```bash
# Both arms
solo robo --calibrate both

# Individual arms
solo robo --calibrate leader
solo robo --calibrate follower
```

### LeRobot Calibration

**Step 1: Find USB Ports**
```bash
lerobot-find-port
```

**Step 2: Setup Motor IDs (first time only)**
```bash
# Follower
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# Leader
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
```

**Step 3: Calibrate**
```bash
# Follower
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower

# Leader
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader
```

### Calibration Process
1. Move arm to neutral position (all joints at midpoint)
2. Press Enter to confirm
3. Move each joint through full range of motion
4. System records MIN/MAX values

### Calibration Storage
Files stored at:
- `~/.cache/huggingface/lerobot/calibration/robots/`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/`

To recalibrate, delete the JSON files and run again.

---

## 5. Teleoperation

### Solo CLI
```bash
solo robo --teleop
solo robo --teleop -y  # Skip prompts
```

### LeRobot
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader
```

### With Camera Display
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --display_data=true
```

---

## 6. Recording Datasets

### Solo CLI
```bash
solo robo --record
```
Follow prompts for dataset name, task description, episode duration, and count.

### LeRobot

```bash
# Setup HuggingFace
huggingface-cli login
export HF_USER=$(huggingface-cli whoami | head -n 1)

# Record
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the object" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10
```

### Recording Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset.num_episodes` | 50 | Number of episodes |
| `--dataset.episode_time_s` | 60 | Episode duration (seconds) |
| `--dataset.reset_time_s` | 60 | Reset duration (seconds) |
| `--dataset.push_to_hub` | true | Upload to HuggingFace |

### Keyboard Controls
- **Right Arrow**: Skip to next episode
- **Left Arrow**: Re-record current episode
- **ESC**: End session and save

### Best Practices
- Record 50+ demonstrations
- Keep movements consistent
- Ensure objects are visible to cameras
- Don't include failed demonstrations

---

## 7. Training Policies

### Solo CLI
```bash
solo robo --train
```
Select policy type (ACT, SmolVLA, Diffusion, etc.) and configure training parameters.

### LeRobot

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=act \
    --output_dir=outputs/train/act_my_task \
    --policy.device=cuda \
    --wandb.enable=true
```

### Training on Apple Silicon
```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=act \
    --policy.device=mps \
    --output_dir=outputs/train/act_my_task
```

### Key Training Parameters

| Parameter | Description |
|-----------|-------------|
| `--policy.type` | act, diffusion, smolvla, etc. |
| `--policy.device` | cuda, cpu, mps |
| `--batch_size` | Batch size (reduce if OOM) |
| `--lr` | Learning rate |
| `--wandb.enable` | Enable W&B logging |

### Upload Trained Model
```bash
huggingface-cli upload ${HF_USER}/my_policy \
    outputs/train/act_my_task/checkpoints/last/pretrained_model
```

---

## 8. Inference

### Solo CLI
```bash
solo robo --inference
```
Provide policy path, task description, and duration.

### LeRobot

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_my_task \
    --dataset.single_task="Pick up the object" \
    --dataset.num_episodes=10 \
    --control.policy.path=${HF_USER}/my_policy
```

### Python API

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Setup
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)}
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0",
    id="my_follower",
    cameras=camera_config
)

robot = SO100Follower(robot_config)
policy = ACTPolicy.from_pretrained("user/my_policy")

robot.connect()

# Inference loop
while True:
    observation = robot.get_observation()
    action = policy.select_action(observation)
    robot.send_action(action)
```

---

## 9. Camera Configuration

### Find Cameras
```bash
lerobot-find-cameras opencv
lerobot-find-cameras realsense  # For Intel RealSense
```

### Camera Types

| Type | Config Class | Use Case |
|------|--------------|----------|
| OpenCV | `OpenCVCameraConfig` | USB webcams |
| RealSense | `RealSenseCameraConfig` | Depth cameras |

### Configuration Parameters

```python
OpenCVCameraConfig(
    index_or_path=0,      # Camera ID or path
    fps=30,               # Frames per second
    width=640,            # Frame width
    height=480,           # Frame height
    color_mode="RGB",     # Color format
)
```

### Multiple Cameras
```bash
--robot.cameras="{
    front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
}"
```

### USB Bandwidth Tips
- Use separate USB buses for multiple cameras
- Reduce FPS or resolution if needed
- Connect cameras directly (avoid hubs)

---

## 10. VLA Models

### Available Models

| Model | Parameters | Training Data | Best For |
|-------|------------|---------------|----------|
| **ACT** | 80M | 50 demos | Quick training, single tasks |
| **SmolVLA** | 450M | Pretrained | Language-conditioned, CPU-friendly |
| **Diffusion** | ~100M | 50+ demos | Complex manipulation |
| **Pi0** | 3.3B | Pretrained | Multi-task, cross-embodiment |
| **GR00T N1.5** | 3B | 20-40 demos | Humanoid robots |

### ACT (Recommended for Hackathon)

Action Chunking with Transformers. Lightweight, data-efficient, fast training.

**Key features:**
- Predicts sequences of actions (chunks) instead of single actions
- Uses CVAE architecture with transformer
- Trains in hours on single GPU
- Works well with 50 demonstrations

**Configuration:**
```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100
```

### SmolVLA

Compact VLA model that runs on CPU.

**Key features:**
- Language-conditioned actions
- 450M parameters
- Pretrained on community datasets
- Supports async inference

**Usage:**
```bash
solo robo --train
# Select: SmolVLA
# Checkpoint: lerobot/smolvla_base
```

### Diffusion Policy

State-of-the-art for complex tasks.

**Configuration:**
```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=diffusion \
    --batch_size=64 \
    --steps=200000
```

---

## 11. Dataset Format

### Directory Structure (v2.1)
```
dataset_repo/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.camera_name/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

### Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `observation.state` | float32 array | Joint positions |
| `observation.images.{camera}` | VideoFrame | Camera images |
| `action` | float32 array | Commanded actions |
| `timestamp` | float32 | Time since episode start |
| `episode_index` | int64 | Episode ID |
| `frame_index` | int64 | Frame within episode |

### Load Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# From HuggingFace Hub
dataset = LeRobotDataset("user/my_dataset")

# Access frames
sample = dataset[0]
print(sample["observation.state"])
print(sample["action"])
```

### Visualize Dataset
```bash
# Online
# Visit: https://huggingface.co/spaces/lerobot/visualize_dataset

# Local
lerobot-dataset-viz --repo-id ${HF_USER}/my_dataset --episode-index 0
```

---

## 12. Troubleshooting

### Motor Issues

**"Missing motor IDs" error:**
```bash
solo robo --motors both
```

**Motors not responding:**
- Check USB connections
- Verify power supply
- Re-run motor setup

### Port Issues

**Permission denied (Linux):**
```bash
sudo chmod 666 /dev/ttyACM*
```

**Port not found:**
```bash
lerobot-find-port
# Disconnect/reconnect USB cables
```

### Camera Issues

**Camera not detected:**
```bash
lerobot-find-cameras opencv
# Try different index_or_path values
```

**Unstable FPS:**
- Reduce resolution or FPS
- Use separate USB buses
- Disable display: `--display_data=false`

### Training Issues

**Out of memory:**
- Reduce batch size: `--batch_size=4`
- Use CPU: `--policy.device=cpu`
- Use gradient checkpointing

**Poor performance:**
- Record more demonstrations
- Ensure consistent demonstrations
- Check camera visibility

### Calibration Issues

**Inconsistent movement:**
- Delete calibration files and recalibrate
- Check for mechanical obstructions

---

## 13. Resources

### Documentation
- Solo Tech: https://docs.getsolo.tech/welcome
- LeRobot: https://huggingface.co/docs/lerobot
- SO-101 Guide: https://huggingface.co/docs/lerobot/so101

### GitHub Repositories
- Solo CLI: https://github.com/GetSoloTech/solo-cli
- LeRobot: https://github.com/huggingface/lerobot
- SO-ARM100 Hardware: https://github.com/TheRobotStudio/SO-ARM100

### Community
- Solo Tech Discord: https://discord.gg/8kR5VvATUq
- LeRobot Discord: https://discord.gg/s3KuuzsPFb

### Cloud GPU (Hackathon)
- Velda Cloud: https://physical-ai.velda.cloud/
- VESSL AI: https://cloud.vessl.ai/

### Pretrained Models
- HuggingFace Hub: https://huggingface.co/lerobot

### Papers
- ACT: https://arxiv.org/abs/2304.13705
- SmolVLA: https://arxiv.org/abs/2506.01844
- Pi0: https://arxiv.org/abs/2410.24164
- GR00T N1.5: https://arxiv.org/abs/2503.14734

### Tutorials
- Robot Learning Tutorial: https://huggingface.co/spaces/lerobot/robot-learning-tutorial
- ACT Training Video: https://www.youtube.com/watch?v=ft73x0LfGpM
