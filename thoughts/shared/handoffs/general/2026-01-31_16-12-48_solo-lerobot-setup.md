---
date: 2026-02-01T00:12:48Z
researcher: Claude
git_commit: 48c426e1c49715f13980cdb6d947b1b25c60fd41
branch: main
repository: fentbot
topic: "Solo CLI LeRobot Setup for Poker Chip Manipulation"
tags: [robotics, lerobot, solo-cli, so101, calibration, hackathon]
status: in_progress
last_updated: 2026-01-31
last_updated_by: Claude
type: implementation_strategy
---

# Handoff: Solo LeRobot SO101 Setup for Poker Chip Manipulation

## Task(s)
Setting up Solo CLI with LeRobot for a SO101 robotic arm to pick up poker chips.

**Status:**
- **Completed:** Python environment setup (3.12), solo-cli installation, arm calibration (leader + follower), camera identification
- **In Progress:** Recording training data with dual cameras
- **Planned:** Train ACT policy on recorded data, run inference

## Critical References
- `/Users/ariasulin/Git/fentbot/docs/QUICKSTART.md` - LeRobot SO-101 quickstart guide
- `/Users/ariasulin/Git/fentbot/solo-cli/solo/commands/robots/lerobot/README.md` - Solo robo docs

## Recent changes
No code changes made - this was a setup/configuration session.

## Learnings

### Environment Setup
- **Python 3.14 is too new** - PyTorch doesn't have wheels for it. Must use Python 3.12:
  ```bash
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -e .
  ```

### Calibration
- Use `solo robo --calibrate all` (not `both` - that's invalid)
- Calibration files stored at:
  - Leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader.json`
  - Follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`
- To reset calibration, delete these files and recalibrate
- **Critical:** During calibration, move each joint through its FULL range of motion. The elbow_flex joint is prone to negative value errors if not moved enough.

### Hardware Ports
- Leader arm: `/dev/tty.usbmodem5A7A0187121` (5V motors)
- Follower arm: `/dev/tty.usbmodem5AB01828691` (12V motors)

### Camera Setup
- **Camera 0** = Side view (right side, perpendicular to robot facing direction)
- **Camera 1** = Top view (behind/above the arm)
- Camera 3 = black/non-functional
- Camera 4 = OBS virtual camera
- Use `lerobot-find-cameras opencv` to detect cameras
- Camera 1 may show low FPS (5fps) due to USB bandwidth - consider lowering resolution to 640x480

### Solo CLI Config
- Config stored at `~/.solo/lerobot_config.json`
- To reset recording settings: `rm -rf ~/.solo/lerobot_config.json`

### Pre-trained Models
- `masato-ka/act_so100_pickandplace_block_circle` - ACT model for pick and place (compatible with SO101)
- Pre-trained models are task-specific; for poker chip task, need to record custom training data

## Artifacts
- `/Users/ariasulin/Git/fentbot/docs/QUICKSTART.md` - Reference documentation
- Camera test images saved to `outputs/captured_images/` and `camera_0.jpg`, `camera_1.jpg`, etc. in solo-cli directory

## Action Items & Next Steps

1. **Complete recording session** with both cameras:
   ```bash
   solo robo --record
   ```
   Configure cameras as: top (index 1), side (index 0), 640x480, 30fps

2. **Record 20-50 episodes** of picking up poker chips:
   - Keep chip position consistent
   - Use smooth, deliberate movements
   - Press Right Arrow to advance episodes
   - Press ESC when done

3. **Train ACT policy:**
   ```bash
   solo robo --train
   ```
   Select ACT policy type, ~100k steps recommended

4. **Run inference:**
   ```bash
   solo robo --inference
   ```

## Other Notes

### Useful Commands
```bash
# Calibrate
solo robo --calibrate all

# Teleoperate (test setup)
solo robo --teleop

# Record training data
solo robo --record

# Train policy
solo robo --train

# Run inference
solo robo --inference

# Find cameras
lerobot-find-cameras opencv

# Motor setup if needed
solo robo --motors all
```

### Direct LeRobot Command (bypass Solo CLI prompts)
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AB01828691 \
    --robot.id=so101_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0187121 \
    --teleop.id=so101_leader \
    --display_data=true \
    --dataset.repo_id=chip_pickup \
    --dataset.single_task="Pick up the poker chip" \
    --dataset.num_episodes=30
```

### Troubleshooting
- Port permission denied (Linux): `sudo chmod 666 /dev/ttyACM*`
- Calibration negative value error: Delete calibration JSON files and recalibrate, ensuring full joint movement
- Camera not found: Check USB connections, try `lerobot-find-cameras opencv`
