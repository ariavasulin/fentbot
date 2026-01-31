---
date: 2026-01-31T19:33:11Z
researcher: Claude
git_commit: cd282893c03bc245c2008f598f5a5f7d6fa50241
branch: main
repository: fentbot
topic: "LeRobot Complete Reference for SO-101 Arms"
tags: [research, lerobot, so101, feetech, act, robotics, imitation-learning]
status: complete
last_updated: 2026-01-31
last_updated_by: Claude
---

# LeRobot Complete Reference for SO-101 Arms

**Date**: 2026-01-31T19:33:11Z
**Researcher**: Claude
**Git Commit**: cd282893c03bc245c2008f598f5a5f7d6fa50241
**Branch**: main
**Repository**: fentbot

## Research Question

Create a full reference document for LeRobot focusing on: Installation and setup for SO-101 arms with Feetech servos, Calibration APIs and programmatic position access, Data recording APIs, ACT policy training, Inference/deployment, and Python APIs for direct robot control.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Calibration APIs](#2-calibration-apis)
3. [Data Recording APIs](#3-data-recording-apis)
4. [ACT Policy Training](#4-act-policy-training)
5. [Inference and Deployment](#5-inference-and-deployment)
6. [Python APIs for Direct Robot Control](#6-python-apis-for-direct-robot-control)

---

## 1. Installation and Setup

### 1.1 System Requirements

- **Python**: >= 3.10 (Python 3.10 recommended)
- **OS**: Linux (Ubuntu), macOS, Windows
- **GPU**: NVIDIA GPU recommended for training (16GB+ VRAM)

### 1.2 Installation Commands

**Option A: Install from PyPI**
```bash
# Core library only
pip install lerobot

# With Feetech motor support (required for SO-101)
pip install 'lerobot[feetech]'

# All features
pip install 'lerobot[all]'
```

**Option B: Install from Source (Recommended for development)**
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

### 1.3 Conda Environment Setup

```bash
# Install miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Create and activate environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install FFmpeg (required for video encoding)
conda install ffmpeg -c conda-forge
```

### 1.4 Linux Build Dependencies

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
```

### 1.5 SO-101 Hardware Bill of Materials

| Component | Quantity | Approx. Cost |
|-----------|----------|--------------|
| STS3215 Servo 7.4V | 7 per arm | ~$97/unit |
| Motor Control Board (Waveshare) | 2 | ~$10.60 each |
| USB-C Cable 2-pack | 1 | ~$7 |
| Power Supply (5V/7.4V) | 2 | ~$10 each |
| Table Clamp | 4 | ~$9 total |
| **Total (2 arms)** | | **~$230 USD** |

### 1.6 Motor Configuration by Arm

**Follower Arm**: 6x STS3215 motors with 1/345 gearing

**Leader Arm** (different gear ratios for easier manipulation):

| Joint | Motor Position | Gear Ratio |
|-------|----------------|------------|
| Base / Shoulder Pan | 1 | 1/191 |
| Shoulder Lift | 2 | 1/345 |
| Elbow Flex | 3 | 1/191 |
| Wrist Flex | 4 | 1/147 |
| Wrist Roll | 5 | 1/147 |
| Gripper | 6 | 1/147 |

### 1.7 USB/Serial Port Setup

**Find Ports (All OSes):**
```bash
lerobot-find-port
```

**Linux Setup:**
```bash
# Check detected ports
ls /dev/ttyACM*

# Grant permissions
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# Or add user to dialout group permanently:
sudo usermod -a -G dialout $USER
# Then log out and back in
```

**Persistent udev Rules** (create `/etc/udev/rules.d/99-lerobot-serial.rules`):
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="YOUR_SERIAL", SYMLINK+="usbserial_lerobot_follower", MODE="0666"
```

**macOS**: Ports appear as `/dev/tty.usbmodemXXXXX`

**Windows**: Ports appear as `COM3`, `COM4`, etc.

### 1.8 Motor ID and Baudrate Setup

**Set up follower arm motors:**
```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0
```

**Set up leader arm motors:**
```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
```

### 1.9 Common Installation Issues

| Issue | Solution |
|-------|----------|
| `COMM_RX_CORRUPT` errors | Ensure unique motor IDs, connect one motor at a time during setup |
| Permission denied on `/dev/ttyACM*` | Run `sudo chmod 666 /dev/ttyACM0` or add user to dialout group |
| FFmpeg/video encoding issues | Install with `conda install ffmpeg=7.1.1 -c conda-forge` |
| Build errors | Install Linux build dependencies listed above |

---

## 2. Calibration APIs

### 2.1 CLI Calibration Commands

**Calibrate follower arm:**
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
```

**Calibrate leader arm:**
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm
```

### 2.2 Calibration Process

1. Move the robot to the position where all joints are in the middle of their ranges
2. Press Enter, then move each joint through its full range of motion
3. The system records MIN, POS (current position), and MAX values for each joint
4. Calibration is saved automatically to JSON file

### 2.3 Calibration File Locations

```
~/.cache/huggingface/lerobot/calibration/
├── robots/
│   └── so101_follower/
│       └── <robot_id>.json
└── teleoperators/
    └── so101_leader/
        └── <teleop_id>.json
```

### 2.4 Calibration JSON Format

```json
{
  "shoulder_pan": {
    "id": 1,
    "drive_mode": 0,
    "homing_offset": 14,
    "range_min": 1015,
    "range_max": 3128
  },
  "shoulder_lift": {
    "id": 2,
    "drive_mode": 0,
    "homing_offset": -1732,
    "range_min": 1499,
    "range_max": 3853
  }
  // ... other joints
}
```

| Field | Description |
|-------|-------------|
| `id` | Motor ID number (1-6) |
| `drive_mode` | 0 = normal, 1 = inverted |
| `homing_offset` | Offset to convert raw position to zero at middle position |
| `range_min` | Minimum raw position value |
| `range_max` | Maximum raw position value |

### 2.5 Python API for Calibration

```python
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_follower_arm",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)  # Connect without loading previous calibration
follower.calibrate()  # Run calibration procedure (saves to JSON automatically)
follower.disconnect()
```

### 2.6 Loading Calibration Programmatically

```python
from pathlib import Path
import json

def load_calibration(robot_id: str) -> dict:
    """Load calibration data from JSON file."""
    fpath = Path(f'~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{robot_id}.json').expanduser()
    with open(fpath) as f:
        return json.load(f)

calibration = load_calibration("my_follower_arm")
```

### 2.7 Reading Current Joint Positions

```python
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(
    port="/dev/ttyACM0",
    motors={
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect(True)

# Read current positions
while True:
    present_pos = bus.sync_read("Present_Position")
    print(present_pos)
    time.sleep(0.02)  # 50 Hz loop
```

### 2.8 Reset Calibration

```bash
# Delete existing calibration files
rm -rf ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/
rm -rf ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/

# Run calibration again
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

---

## 3. Data Recording APIs

### 3.1 CLI Recording Command

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/my-dataset \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick up the cube"
```

### 3.2 Recording Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset.num_episodes` | 50 | Total episodes to record |
| `--dataset.episode_time_s` | 60 | Duration per episode (seconds) |
| `--dataset.reset_time_s` | 60 | Reset time between episodes |
| `--dataset.single_task` | - | Task description string |
| `--dataset.repo_id` | - | HuggingFace repo ID |
| `--dataset.push_to_hub` | true | Auto-upload after recording |
| `--fps` | 30 | Target frames per second |
| `--display_data` | false | Show Rerun visualization |

### 3.3 Python Recording API

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so_leader.so100_leader import SO100Leader
from lerobot.scripts.lerobot_record import record_loop

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60

# Create robot configuration with cameras
robot_config = SO100FollowerConfig(
    id="my_follower_arm",
    cameras={
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)
    },
    port="/dev/ttyACM0",
)

teleop_config = SO100LeaderConfig(
    id="my_leader_arm",
    port="/dev/ttyACM1",
)

# Initialize robot and teleoperator
robot = SO100Follower(robot_config)
teleop = SO100Leader(teleop_config)

# Configure dataset features from robot
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="username/dataset-name",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Connect devices
robot.connect()
teleop.connect()

# Recording loop
for episode_idx in range(NUM_EPISODES):
    record_loop(
        robot=robot,
        teleop=teleop,
        fps=FPS,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task="Pick up the cube",
        display_data=True,
    )
    dataset.save_episode()

# CRITICAL: Finalize and upload
dataset.finalize()  # Must call before push_to_hub!
dataset.push_to_hub()
robot.disconnect()
```

### 3.4 LeRobotDataset v3.0 Format

**Directory Layout:**
```
dataset_repo/
├── meta/
│   ├── info.json           # Schema, fps, features
│   ├── stats.json          # Normalization statistics
│   ├── tasks.jsonl         # Task descriptions
│   └── episodes/           # Per-episode metadata
├── data/
│   └── chunk-000/          # Tabular data (Parquet)
└── videos/
    └── front/              # Video files per camera
        └── chunk-000/
```

**info.json Schema:**
```json
{
  "codebase_version": "v3.0",
  "robot_type": "so101",
  "fps": 30,
  "total_episodes": 50,
  "total_frames": 3000,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [7]
    },
    "observation.images.front": {
      "dtype": "video",
      "shape": [480, 640, 3]
    },
    "action": {
      "dtype": "float32",
      "shape": [7]
    }
  }
}
```

### 3.5 Loading Existing Datasets

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load from Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access single sample
sample = dataset[100]
# Returns: {
#   'observation.state': tensor([...]),
#   'action': tensor([...]),
#   'observation.images.front': tensor([C, H, W]),
#   'timestamp': tensor(1.234),
# }
```

### 3.6 Camera Configuration

**Multi-camera setup:**
```bash
--robot.cameras="{
  front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
  side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}
}"
```

**Find available cameras:**
```bash
lerobot-find-cameras opencv
lerobot-find-cameras realsense
```

### 3.7 Replay Recorded Episodes

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --dataset.repo_id=${HF_USER}/my-dataset \
  --dataset.episode=0
```

### 3.8 Dataset Visualization

```bash
# Local visualization
lerobot-dataset-viz \
  --repo-id lerobot/pusht \
  --episode-index 0

# Online: https://huggingface.co/spaces/lerobot/visualize_dataset
```

---

## 4. ACT Policy Training

### 4.1 ACT Architecture Overview

ACT (Action Chunking with Transformers) predicts chunks of future actions rather than single actions, reducing compounding errors.

**Key Components:**
- **Vision Backbone**: ResNet-18 processes camera images
- **Transformer Encoder**: Synthesizes camera features, joint positions, and latent variable
- **Transformer Decoder**: Generates action sequences

**Key Concepts:**
- **Action Chunking**: Predicts k actions at once (typically 50-100)
- **Temporal Ensembling**: Overlapping chunks aggregated for smooth motion
- **Latent Variable (CVAE)**: Encodes trajectory diversity

### 4.2 Training CLI Command

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

### 4.3 ACT Configuration Parameters

**Input/Output:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_obs_steps` | 1 | Number of observation steps |
| `chunk_size` | 100 | Action prediction chunk size |
| `n_action_steps` | 100 | Actions executed per invocation |

**Vision Backbone:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vision_backbone` | "resnet18" | Torchvision ResNet variant |
| `pretrained_backbone_weights` | "ResNet18_Weights.IMAGENET1K_V1" | Initialization weights |

**Transformer Architecture:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim_model` | 512 | Hidden dimension |
| `n_heads` | 8 | Attention heads |
| `dim_feedforward` | 3200 | Feed-forward dimension |
| `n_encoder_layers` | 4 | Encoder layers |
| `n_decoder_layers` | 1 | Decoder layers |
| `dropout` | 0.1 | Dropout rate |

**VAE Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vae` | True | Enable variational objective |
| `latent_dim` | 32 | Latent space dimension |
| `kl_weight` | 10.0 | KL-divergence weight (β) |

**Optimization:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_lr` | 1e-5 | Learning rate |
| `optimizer_weight_decay` | 1e-4 | Weight decay |

### 4.4 Memory-Optimized Training

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.dim_model=256 \
  --batch_size=8 \
  --policy.device=cuda
```

### 4.5 Multi-GPU Training

```bash
pip install accelerate

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=fp16 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_multi_gpu
```

**Learning Rate Scaling (Manual):**
```bash
# Base LR: 1e-4, with 2 GPUs -> 2e-4
accelerate launch --num_processes=2 $(which lerobot-train) \
  --optimizer.lr=2e-4 \
  --dataset.repo_id=lerobot/pusht \
  --policy=act
```

### 4.6 Checkpointing and Resuming

**Checkpoint Location:**
```
outputs/train/<job_name>/checkpoints/
├── 010000/pretrained_model/
├── 020000/pretrained_model/
└── last/pretrained_model/
```

**Resume Training:**
```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

**Upload Checkpoint to Hub:**
```bash
huggingface-cli upload ${HF_USER}/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

### 4.7 Training Best Practices

| Recommendation | Details |
|----------------|---------|
| Quality over Quantity | Collect clean, consistent demonstrations |
| Camera Placement | Use front + top-down view; keep cameras fixed |
| Batch Size | Start with 8; use 16+ for stable gradients |
| Training Duration | Expect 100k steps in a few hours on single GPU |
| Chunk Size | Use 50 for small datasets |

### 4.8 Expected Results

- 50 episodes with 100k training steps: ~70% success rate on simple pick-and-place
- Training time: ~1.5-2 hours on NVIDIA A100 for 100k steps
- Limited generalization to significantly different object positions

---

## 5. Inference and Deployment

### 5.1 Loading Trained Policies

**From Hugging Face Hub:**
```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
```

**From Local Checkpoint:**
```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("outputs/train/act_so101_test/checkpoints/last/pretrained_model")
```

### 5.2 CLI Inference Command

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Pick up the cube" \
  --policy.path=${HF_USER}/act_so101_test
```

### 5.3 Simulation Evaluation

```bash
lerobot-eval \
  --policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --device=cuda
```

### 5.4 Python Inference API

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.processor import make_default_processors

# Load policy
policy = ACTPolicy.from_pretrained("${HF_USER}/act_so101_test")
policy = policy.to("cuda").eval()

# Setup preprocessors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path="${HF_USER}/act_so101_test",
    dataset_stats=dataset.meta.stats,
)

# Inference loop
robot.connect()
while True:
    obs = robot.get_observation()
    action = predict_action(
        observation=obs,
        policy=policy,
        device="cuda",
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task="Pick up the cube",
    )
    robot.send_action(action)
```

### 5.5 Asynchronous Inference (Network Deployment)

**Start Policy Server:**
```bash
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080
```

**Start Robot Client:**
```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_so100 \
    --task="Pick up the cube" \
    --policy_type=act \
    --pretrained_name_or_path=lerobot/act_aloha \
    --policy_device=cuda
```

### 5.6 Latency Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actions_per_chunk` | 50 | Actions output at once |
| `chunk_size_threshold` | 0.7 | Queue threshold for new observation |

**Optimization Strategies:**
- Choose lighter policies (SmolVLA ~2GB) over heavy ones (PI0 ~14GB)
- Reduce FPS if running out of actions in queue
- Tune `chunk_size_threshold` (0.5-0.6 works well)

### 5.7 Hardware Deployment Platforms

| Platform | Use Case | Notes |
|----------|----------|-------|
| NVIDIA GPU (Desktop) | Training + Inference | Full capability |
| Apple Silicon (MPS) | Training + Inference | `--policy.device=mps` |
| NVIDIA Jetson Orin | Edge Inference + Training | Recommended for real robots |
| Raspberry Pi | Lightweight ACT only | Async server needed for VLAs |

### 5.8 Safety Features

**max_relative_target Parameter:**
```bash
# With safety limit (default 5 degrees)
--robot.max_relative_target=5.0

# Without limit (experienced users)
--robot.max_relative_target=null
```

---

## 6. Python APIs for Direct Robot Control

### 6.1 Core Classes

```python
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.motors import Motor, MotorNormMode
```

### 6.2 Reading Joint States

**Using ManipulatorRobot:**
```python
robot = ManipulatorRobot(config)
robot.connect()
obs = robot.get_observation()  # Returns dict with "observation.state" tensor
```

**Using Motor Bus Directly:**
```python
bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={...})
bus.connect()

# Read current position
position = bus.read("Present_Position")  # numpy array

# Read velocity
velocity = bus.read("Present_Speed")

# Read load/current
load = bus.read("Present_Load")
```

### 6.3 Commanding Joint Positions

**High-level API:**
```python
action = torch.tensor([0.0, 45.0, -30.0, 60.0, 0.0, 50.0])  # degrees
actual_action = robot.send_action(action)
```

**Motor Bus Direct Control:**
```python
goal = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0
}
bus.sync_write("Goal_Position", goal)
```

### 6.4 Torque Control

```python
from lerobot.common.robot_devices.motors.feetech import OperatingMode

# Enable torque (lock position)
bus.write("Torque_Enable", TorqueMode.ENABLED.value)

# Disable torque (allow manual movement)
bus.write("Torque_Enable", TorqueMode.DISABLED.value)

# Context manager for safe torque operations
with bus.torque_disabled():
    bus.configure_motors()
```

### 6.5 Camera Integration

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode

config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
)

camera = OpenCVCamera(config)
camera.connect()

# Synchronous read
frame = camera.read()

# Asynchronous read (background thread)
frame = camera.async_read(timeout_ms=200)

camera.disconnect()
```

### 6.6 Complete Control Example

```python
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode
import time

# ===== MOTOR SETUP =====
bus = FeetechMotorsBus(
    port="/dev/ttyUSB0",
    motors={
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
)

# ===== CAMERA SETUP =====
cam_config = OpenCVCameraConfig(
    index_or_path=0, fps=30, width=640, height=480, color_mode=ColorMode.RGB
)
camera = OpenCVCamera(cam_config)

# ===== CONNECT =====
bus.connect(True)
camera.connect()

# ===== CONFIGURE MOTORS =====
with bus.torque_disabled():
    bus.configure_motors()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", motor, 16)

# ===== CONTROL LOOP =====
try:
    fps = 30
    while True:
        start_time = time.time()

        # Read current state
        positions = bus.sync_read("Present_Position")
        print(f"Current positions: {positions}")

        # Capture camera frame
        frame = camera.async_read(timeout_ms=200)

        # Send goal position
        goal = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 50.0
        }
        bus.sync_write("Goal_Position", goal)

        # Maintain loop frequency
        elapsed = time.time() - start_time
        time.sleep(max(1/fps - elapsed, 0))

except KeyboardInterrupt:
    print("Stopping...")
finally:
    bus.disconnect()
    camera.disconnect()
```

### 6.7 Thread Safety Note

LeRobot does NOT implement explicit thread safety. For multi-threaded access:

```python
import threading

class ThreadSafeRobot:
    def __init__(self, robot):
        self.robot = robot
        self.lock = threading.Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action):
        with self.lock:
            return self.robot.send_action(action)
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Find ports | `lerobot-find-port` |
| Setup motors | `lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0` |
| Calibrate | `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_robot` |
| Teleoperate | `lerobot-teleoperate --robot.type=so101_follower --teleop.type=so101_leader ...` |
| Record | `lerobot-record --dataset.repo_id=${HF_USER}/dataset --dataset.num_episodes=10 ...` |
| Train | `lerobot-train --dataset.repo_id=${HF_USER}/dataset --policy.type=act ...` |
| Evaluate | `lerobot-eval --policy.path=${HF_USER}/policy --env.type=pusht ...` |
| Replay | `lerobot-replay --dataset.repo_id=${HF_USER}/dataset --dataset.episode=0 ...` |

---

## Additional Resources

- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/en/index)
- [SO-ARM100 Hardware Repository](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot Discord Community](https://discord.com/invite/s3KuuzsPFb)
- [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
- [Waveshare SO-ARM Wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control)
