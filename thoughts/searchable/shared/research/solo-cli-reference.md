# Solo CLI Reference Guide

**Date**: 2026-01-31
**Repository**: `solo-cli/`
**Framework**: Typer-based CLI for robotics using LeRobot

---

## Overview

Solo CLI is a command-line interface tool for controlling robotic arms through the LeRobot framework. It provides a unified interface for robot calibration, teleoperation, dataset recording, policy training, and inference.

## Architecture

```
solo-cli/
├── solo/
│   ├── __init__.py          # Package version (__version__ = "0.1.0")
│   ├── main.py               # Entry point (calls cli.py main())
│   ├── cli.py                # Main Typer app with command definitions
│   ├── config/               # Configuration system
│   │   ├── __init__.py       # CONFIG_DIR (~/.solo), CONFIG_PATH
│   │   ├── config_loader.py  # YAML config loading utilities
│   │   └── config.yaml       # Server configurations (vLLM, Ollama, llama.cpp)
│   ├── commands/             # Command implementations
│   │   ├── robo.py           # Robot command handler
│   │   ├── serve.py          # LLM server management
│   │   ├── status.py         # Server status checking
│   │   ├── stop.py           # Server shutdown
│   │   ├── download_hf.py    # HuggingFace model download
│   │   ├── models_list.py    # Available models listing
│   │   ├── benchmark.py      # LLM benchmarking
│   │   ├── finetune.py       # Model fine-tuning
│   │   ├── setup_usb.py      # USB permissions setup (Linux)
│   │   ├── test.py           # Testing utilities
│   │   └── robots/           # Robot-specific implementations
│   │       └── lerobot/      # LeRobot integration
│   └── utils/                # Utility modules
│       ├── hardware.py       # GPU/system detection
│       ├── server_utils.py   # Server management utilities
│       ├── docker_utils.py   # Docker container management
│       ├── hf_utils.py       # HuggingFace utilities
│       ├── llama_cpp_utils.py # llama-cpp-python setup
│       └── nvidia.py         # NVIDIA CUDA detection
└── setup.py                  # Package installation
```

---

## CLI Commands

### Entry Point: `solo/cli.py`

The main Typer application defines the following commands:

| Command | Description | Implementation |
|---------|-------------|----------------|
| `solo serve` | Start LLM inference server | `solo/commands/serve.py` |
| `solo stop` | Stop running server | `solo/commands/stop.py` |
| `solo status` | Check server status | `solo/commands/status.py` |
| `solo download` | Download HuggingFace model | `solo/commands/download_hf.py` |
| `solo models` | List available models | `solo/commands/models_list.py` |
| `solo benchmark` | Run LLM benchmarks | `solo/commands/benchmark.py` |
| `solo finetune` | Fine-tune models | `solo/commands/finetune.py` |
| `solo setup-usb` | Configure USB permissions | `solo/commands/setup_usb.py` |
| `solo test` | Run tests | `solo/commands/test.py` |
| `solo robo` | Robot control | `solo/commands/robo.py` |

---

## Robot Control: `solo robo`

### Location: `solo/commands/robo.py` → `solo/commands/robots/lerobot/`

### Supported Robot Types

| Robot Type | Motors | Leader/Follower | Port Type |
|------------|--------|-----------------|-----------|
| `so100` | Feetech STS3215 | Both use same motors | USB Serial |
| `so101` | Feetech STS3215 | Both use same motors | USB Serial |
| `koch` | Dynamixel XL330/XL430 | Different motors per arm | USB Serial |
| `bi_so100` | Feetech STS3215 | Bimanual (2 leaders, 2 followers) | USB Serial |
| `bi_so101` | Feetech STS3215 | Bimanual (2 leaders, 2 followers) | USB Serial |
| `realman_r1d2` | RealMan + SO101 leader | Hybrid (Network + USB) | IP:Port + USB |
| `realman_rm65` | RealMan RM65 | Hybrid | IP:Port + USB |
| `realman_rm75` | RealMan RM75 (7 DOF) | Hybrid | IP:Port + USB |

### Command Flags

```bash
solo robo [OPTIONS]

Options:
  --calibrate        Calibrate robot arms
  --motor-setup      Setup motor IDs and homing offsets
  --teleop           Start teleoperation mode
  --record           Record demonstration dataset
  --train            Train a policy
  --inference        Run policy inference
  --replay           Replay recorded episodes
  --robot-type TEXT  Robot type (so100, so101, koch, bi_so100, bi_so101, realman_r1d2)
  --scan             Scan for connected motors
  -y, --yes          Use preconfigured settings without prompts
```

---

## LeRobot Integration

### File Structure: `solo/commands/robots/lerobot/`

| File | Purpose |
|------|---------|
| `lerobot.py` | Main dispatcher - routes to appropriate mode handlers |
| `calibration.py` | Arm calibration workflows |
| `teleoperation.py` | Leader-follower teleoperation |
| `config.py` | Robot configuration classes and utilities |
| `ports.py` | Serial port detection and management |
| `scan.py` | Motor scanning (Dynamixel + Feetech) |
| `cameras.py` | Camera detection and configuration |
| `dataset.py` | Dataset management utilities |
| `auth.py` | HuggingFace authentication |
| `mode_config.py` | Per-mode configuration persistence |
| `realman_config.py` | RealMan-specific configuration |
| `modes/recording.py` | Dataset recording workflow |
| `modes/training.py` | Policy training workflow |
| `modes/inference.py` | Policy inference workflow |

---

## Modes

### 1. Calibration (`calibration.py`)

Calibrates motor positions for robot arms.

**Key Functions:**
- `run_calibration(robot_type, arm_type, port, arm_id)` - Main calibration entry
- `calibrate_single_arm(...)` - Single arm calibration
- `calibrate_bimanual_arm(...)` - Bimanual arm calibration
- `calibrate_realman_robot(...)` - RealMan hybrid calibration

**Workflow:**
1. Detect or select arm type (leader/follower)
2. Detect or input port
3. Create robot configuration
4. Run `lerobot.calibrate()` with appropriate config

### 2. Motor Setup (via `lerobot.py`)

Sets up motor IDs using LeRobot's `setup_motors` command.

### 3. Teleoperation (`teleoperation.py`)

Leader-follower teleoperation where human moves leader arm and follower mirrors.

**Key Functions:**
- `run_teleoperation(robot_type, leader_port, follower_port, camera_config, ...)` - Main entry
- `perform_teleop_with_retry(...)` - Handles connection retries
- `build_calibration_paths(...)` - Locates calibration files

**Configuration Classes (from `config.py`):**
- `SO100LeaderConfig`, `SO100FollowerConfig`
- `SO101LeaderConfig`, `SO101FollowerConfig`
- `KochLeaderConfig`, `KochFollowerConfig`
- `BiSO100LeaderConfig`, `BiSO100FollowerConfig`
- `BiSO101LeaderConfig`, `BiSO101FollowerConfig`
- `RealManFollowerConfig` (with SO101 leader)

### 4. Recording (`modes/recording.py`)

Records demonstration datasets for imitation learning.

**Key Functions:**
- `run_recording(config, robot_type, ...)` - Main recording entry
- `setup_recording_args(...)` - Interactive argument setup
- `execute_recording(...)` - Executes LeRobot recording

**Parameters:**
- `dataset_repo_id` - HuggingFace repository ID (format: `owner/dataset_name`)
- `task_description` - Natural language task description
- `episode_time` - Duration per episode (default: 60s)
- `num_episodes` - Number of episodes to record (default: 50)
- `fps` - Recording framerate (default: 30)
- `push_to_hub` - Auto-push to HuggingFace Hub
- `video_backend` - `torchcodec` (default) or `pyav`

**Resume Support:**
- Checks for existing datasets in `HF_LEROBOT_HOME / repo_id`
- Validates `meta/info.json` exists for valid datasets
- Handles incomplete datasets (missing metadata)

### 5. Training (`modes/training.py`)

Trains policies from recorded datasets.

**Key Functions:**
- `run_training(config)` - Main training entry
- `setup_training_args(config)` - Interactive setup
- `execute_training(training_args)` - Executes training
- `list_local_models(output_dir)` - Lists trained models

**Supported Policy Types:**
| Policy | Config Path | Description |
|--------|-------------|-------------|
| `smolvla` | `smolvla/smolvla_libero_gpu.yaml` | Vision-Language-Action model |
| `act` | `act/act_koch_real.yaml` | Action Chunking Transformer |
| `pi0` | `pi0/pi0.yaml` | Pi-Zero policy |
| `tdmpc` | `tdmpc/default.yaml` | TD-MPC algorithm |
| `diffusion` | `diffusion/default.yaml` | Diffusion Policy |

**Training Parameters:**
- `dataset_repo_id` - Dataset to train on
- `policy_type` - One of the supported policies
- `output_dir` - Model output directory (default: `outputs/train/`)
- `training_steps` - Number of steps (default: 10000)
- `batch_size` - Training batch size (default: 8)
- `learning_rate` - Learning rate (default: 1e-5)
- `grad_accum_steps` - Gradient accumulation (default: 1)
- `use_wandb` - Enable Weights & Biases logging
- `wandb_project` - W&B project name (default: `lerobot`)

**Dataset Handling:**
- Supports local paths: `local/dataset_name`
- Supports HuggingFace Hub: `username/dataset_name`
- Validates dataset existence before training

### 6. Inference (`modes/inference.py`)

Runs trained policies on the robot.

**Key Functions:**
- `run_inference(config, robot_type, ...)` - Main inference entry
- `setup_inference_args(config, robot_type, ...)` - Interactive setup
- `execute_inference(inference_args)` - Executes inference

**Parameters:**
- `policy_path` - Path to trained policy checkpoint
- `inference_time` - Duration to run (default: 30s)
- `fps` - Inference framerate (default: 30)
- `use_teleoperation` - Allow human intervention during inference
- `warmup_time` - Robot warmup duration (default: 2s)

**Model Discovery:**
- Auto-detects local models in `outputs/train/`
- Lists available checkpoints for selection

### 7. Replay (via `lerobot.py`)

Replays recorded episodes without a policy.

**Parameters:**
- `dataset_repo_id` - Dataset to replay
- `episode` - Specific episode number
- `fps` - Playback framerate
- `play_sounds` - Audio feedback

---

## Port Detection and Motor Scanning

### Location: `scan.py`, `ports.py`

### Serial Port Detection

**Platform-specific patterns:**

| Platform | Patterns |
|----------|----------|
| Windows | COM ports via pyserial (excludes Bluetooth) |
| macOS | `/dev/tty.usbmodem*`, `/dev/tty.usbserial*`, `/dev/cu.*` |
| Linux | `/dev/ttyACM*`, `/dev/ttyUSB*` |

### Motor Scanning

**Dynamixel (Koch arms):**
- Protocol 2.0
- Baudrate: 1,000,000
- Scans motor IDs 1-20
- Model detection: XL330-M077 (leader), XL430-W250/XL330-M288 (follower)

**Feetech (SO100/SO101 arms):**
- Protocol 0 (STS/SMS series) or Protocol 1 (SCS series)
- Baudrate: 1,000,000
- Model detection: STS3215 (model 777), STS3250 (model 2825)

### Arm Type Detection

**Koch (Dynamixel):** Distinguished by motor model numbers
- Leader: XL330-M077 only (model 1190)
- Follower: XL430 (1060), XL330-M288 (1200), etc.

**SO100/SO101 (Feetech):** Distinguished by voltage
- Leader: ~5V (4.5-6.5V range)
- Follower: ~12V (10-14V range)

---

## Camera Support

### Location: `cameras.py`, `config.py`

### Supported Camera Types

| Type | Library | Detection |
|------|---------|-----------|
| OpenCV | `cv2` | Index-based (0, 1, 2...) |
| RealSense | `pyrealsense2` | Serial number |

### Camera Configuration

```python
camera_config = {
    'enabled': True,
    'cameras': [
        {
            'camera_id': 0,
            'camera_type': 'OpenCV',
            'angle': 'front',  # front, top, side, wrist
            'camera_info': {...}
        }
    ]
}
```

### FPS Normalization

Requested FPS is normalized to common values:
- ≥55 FPS → 60 FPS
- <55 FPS → 30 FPS (default)

---

## Configuration System

### Location: `solo/config/`

### Main Config File: `~/.solo/config.json`

```json
{
    "server": {
        "type": "lerobot"
    },
    "lerobot": {
        "robot_type": "so101",
        "leader_port": "/dev/ttyACM0",
        "follower_port": "/dev/ttyACM1",
        "leader_calibrated": true,
        "follower_calibrated": true,
        "known_ids_by_type": {
            "so101": {
                "leaders": ["my_leader"],
                "followers": ["my_follower"]
            }
        },
        "mode_configs": {
            "teleop": {...},
            "recording": {...},
            "training": {...},
            "inference": {...},
            "replay": {...}
        }
    },
    "hugging_face": {
        "username": "your_username"
    }
}
```

### Server Config: `solo/config/config.yaml`

Defines LLM server configurations for:
- **vLLM**: Default model `unsloth/Llama-3.2-1B-Instruct`, port 5070
- **Ollama**: Default model `llama3.2:1b`, port 5070
- **llama.cpp**: Default model with Q4_K_M quantization, GPU-specific CMAKE_ARGS

---

## RealMan Robot Support

### Location: `realman_config.py`

### Configuration

```yaml
# ~/.solo/realman_config.yaml
robot:
  ip: '192.168.1.18'
  port: 8080
  model: 'R1D2'

control:
  mode: 'cartesian'
  update_rate: 80

safety:
  collision_level: 3
  enable_deadman: true
  enable_collision_detection: true
  min_z_position: null  # Optional Z safety limit
  z_limit_action: 'clamp'

limits:
  max_joint_velocity: 30.0
```

### Supported Models

| Model | DOF |
|-------|-----|
| R1D2 | 6 |
| RM65 | 6 |
| RM75 | 7 |
| RML63 | 6 |
| ECO65 | 6 |
| GEN72 | 7 |

### Hybrid Setup

RealMan robots use a hybrid configuration:
- **Leader**: SO101 arm (USB serial) for teleoperation input
- **Follower**: RealMan arm (network) mirrors leader movements

---

## HuggingFace Integration

### Location: `auth.py`, `hf_utils.py`

### Authentication

Uses `huggingface_hub` library:
- Token stored via `hf auth login`
- Username saved to `~/.solo/config.json`
- `whoami()` API for verification

### Model Utilities (`hf_utils.py`)

**Functions:**
- `get_available_models(repo_id, suffix)` - List models in repo
- `select_best_model_file(models)` - Select best quantization

**Quantization Priority:**
1. `q4_k_m` (preferred)
2. `q5_k_m`
3. `q4_k_s`
4. `q8_0`
5. First available

---

## LLM Server Support

### Location: `serve.py`, `utils/server_utils.py`, `utils/docker_utils.py`

### Supported Servers

| Server | Container Image | Native Support |
|--------|-----------------|----------------|
| vLLM | `vllm/vllm-openai` | No |
| Ollama | `ollama/ollama` | Yes |
| llama.cpp | N/A (pip install) | Yes |

### GPU Detection

**`utils/hardware.py`:**
- NVIDIA: Checks `nvidia-smi`
- AMD: Checks `rocminfo`
- Apple Silicon: Checks `platform.processor()`

### llama-cpp-python Setup (`utils/llama_cpp_utils.py`)

CMAKE_ARGS by platform:
- NVIDIA: `-DGGML_CUDA=on`
- AMD: `-DGGML_HIPBLAS=on`
- Apple Silicon: `-DGGML_METAL=on`

---

## Key Design Patterns

### 1. Lazy Imports

Heavy imports (LeRobot, PyTorch) are deferred until actually needed:

```python
def run_calibration(...):
    from lerobot.scripts.lerobot_control import calibrate  # Lazy
    calibrate(...)
```

### 2. Configuration Persistence

Mode-specific settings are saved after successful operations:
- `save_teleop_config()`
- `save_recording_config()`
- `save_training_config()`
- `save_inference_config()`
- `save_replay_config()`

Use `-y` flag to automatically reuse saved settings.

### 3. Port Retry Logic

Failed connections trigger automatic port re-detection:
- `perform_teleop_with_retry()` catches `PortNotFoundError`
- `detect_and_retry_ports()` updates all mode configs with new ports

### 4. Known ID Tracking

Arm IDs are tracked per robot type for quick selection:

```python
known_ids_by_type = {
    "so101": {
        "leaders": ["leader_1", "leader_2"],
        "followers": ["follower_1"]
    }
}
```

### 5. Robot Type Inference

Can infer robot type from:
- Arm ID name patterns (`so101_leader`, `koch_follower`)
- Connected motor models
- Motor voltage (5V=leader, 12V=follower for Feetech)

---

## Common Workflows

### 1. First-Time Setup

```bash
# 1. Scan for motors
solo robo --scan

# 2. Calibrate leader arm
solo robo --calibrate --robot-type so101

# 3. Calibrate follower arm
solo robo --calibrate --robot-type so101

# 4. Test teleoperation
solo robo --teleop
```

### 2. Record Dataset

```bash
# Interactive recording
solo robo --record

# With preconfigured settings
solo robo --record -y
```

### 3. Train Policy

```bash
# Interactive training
solo robo --train

# Trains on dataset, outputs to outputs/train/
```

### 4. Deploy Policy

```bash
# Run inference with trained policy
solo robo --inference

# With teleoperation fallback
solo robo --inference  # Select "use_teleoperation: true"
```

---

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `PortNotFoundError` | Port changed or disconnected | Re-run with port detection |
| `CalibrationError` | Arm not calibrated | Run `--calibrate` first |
| `dynamixel_sdk not installed` | Missing motor SDK | `pip install dynamixel-sdk` |
| `scservo_sdk not installed` | Missing Feetech SDK | `pip install feetech-servo-sdk` |

### Diagnostics

```bash
# Detailed motor diagnostics
solo robo --scan

# Check specific port
# Uses diagnose_connection() from scan.py
```

---

## Dependencies

### Core
- `typer` - CLI framework
- `rich` - Terminal formatting
- `pyyaml` - YAML parsing
- `psutil` - Process management

### Robotics
- `lerobot` - Core robotics framework
- `dynamixel-sdk` - Dynamixel motor control
- `feetech-servo-sdk` (scservo_sdk) - Feetech motor control
- `pyserial` - Serial port access

### Vision
- `opencv-python` - Camera access
- `pyrealsense2` - RealSense cameras

### ML/Training
- `torch` - PyTorch
- `wandb` - Experiment tracking
- `huggingface_hub` - Model/dataset hosting

### LLM Serving
- `llama-cpp-python` - Local LLM inference
- `docker` - Container management

---

## File Locations Summary

| Purpose | Path |
|---------|------|
| Config directory | `~/.solo/` |
| Main config | `~/.solo/config.json` |
| Calibration files | `~/.cache/huggingface/lerobot/calibration/{robot_type}/{arm_id}.json` |
| Datasets | `~/.cache/huggingface/lerobot/{repo_id}/` |
| Trained models | `outputs/train/{policy}_{timestamp}/` |
| RealMan config | `~/.solo/realman_config.yaml` or `./robot_config.yaml` |
