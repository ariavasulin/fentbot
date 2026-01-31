# Vision-Language-Action (VLA) Models Reference

## Physical AI Hackathon - Model Selection Guide

---

## Quick Comparison

| Model | Size | Data Needed | Language | Hardware | Best For |
|-------|------|-------------|----------|----------|----------|
| **ACT** | 80M | 50 demos | No | Any GPU | Beginners, fast iteration |
| **SmolVLA** | 450M | Pretrained | Yes | CPU/GPU | Budget, language tasks |
| **Diffusion** | ~100M | 50+ demos | No | GPU | Complex manipulation |
| **Pi0** | 3.3B | Pretrained | Yes | GPU | Multi-task generalization |
| **Pi0-FAST** | 3.3B | Pretrained | Yes | GPU | Better language following |
| **GR00T N1.5** | 3B | 20-40 demos | Yes | GPU | Humanoid robots |
| **OpenVLA** | 7B | Pretrained | Yes | 16GB+ GPU | General manipulation |

**Hackathon Recommendation**: Start with **ACT** for fastest results, try **SmolVLA** if you need language conditioning.

---

## ACT (Action Chunking with Transformers)

### Overview
Lightweight imitation learning algorithm that predicts sequences of actions (chunks) instead of single actions. Created by Tony Zhao et al. for the ALOHA system.

### Key Concepts

**Action Chunking**: Instead of predicting one action per timestep, ACT predicts k actions at once.
- Reduces compounding errors
- Handles temporal inconsistencies in demonstrations
- Creates smoother trajectories

**Temporal Ensembling**: Combines overlapping action chunks for smoother execution.

### Architecture
- **Vision**: ResNet-18 backbone
- **Encoder**: Transformer for synthesis
- **Decoder**: Transformer for action generation
- **Method**: Conditional VAE (CVAE)

### Configuration

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10 \
    --lr=1e-5 \
    --batch_size=8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 100 | Actions per prediction |
| `n_action_steps` | 100 | Steps to execute |
| `kl_weight` | 10 | CVAE regularization |

### Training Requirements
- **Data**: 50 demonstrations
- **Time**: ~5 hours on RTX 2080 Ti for 100k steps
- **Inference**: ~0.01 seconds

### Resources
- Paper: https://arxiv.org/abs/2304.13705
- LeRobot Docs: https://huggingface.co/docs/lerobot/en/act
- ALOHA Project: https://tonyzhaozh.github.io/aloha/

---

## SmolVLA (Hugging Face)

### Overview
Compact, open-source VLA designed for affordability. Runs on CPU, single GPU, or MacBook. 450M parameters.

### Key Features
- Language-conditioned actions
- Pretrained on 10M frames from 487 community datasets
- 87.3% success rate on LIBERO benchmark
- Supports asynchronous inference (30% faster)

### Architecture
- **VLM**: SmolVLM2 (SigLIP vision + SmolLM2 language)
- **Action Expert**: 100M parameter transformer

### Usage

```bash
# With Solo CLI
solo robo --train
# Select: SmolVLA
# Checkpoint: lerobot/smolvla_base

# With LeRobot
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base
```

### Pretrained Model
- HuggingFace: `lerobot/smolvla_base`

### Resources
- Blog: https://huggingface.co/blog/smolvla
- Paper: https://arxiv.org/abs/2506.01844

---

## Diffusion Policy

### Overview
Uses diffusion models to generate action sequences. State-of-the-art for complex manipulation tasks.

### Key Features
- Handles multi-modal action distributions
- Better for complex, contact-rich tasks
- Longer training time than ACT

### Configuration

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.type=diffusion \
    --batch_size=64 \
    --steps=200000
```

### Training Requirements
- **Data**: 50+ demonstrations
- **Time**: Longer than ACT
- **GPU**: Recommended for training

---

## Pi0 (Physical Intelligence)

### Overview
3.3B parameter VLA for general robot control. Uses flow matching for smooth 50Hz action trajectories.

### Key Features
- Trained on 7 robotic platforms, 68 tasks
- Cross-embodiment generalization
- Tasks: laundry folding, table bussing, grocery bagging

### Variants

| Variant | Description |
|---------|-------------|
| **Pi0** | Base model, flow matching |
| **Pi0-FAST** | Autoregressive, better language |
| **Pi0.5** | Open-world generalization |

### Installation

```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```

### Resources
- Pi0 Paper: https://arxiv.org/abs/2410.24164
- Pi0.5 Paper: https://arxiv.org/abs/2504.16054
- OpenPi GitHub: https://github.com/Physical-Intelligence/openpi

---

## GR00T N1.5 (NVIDIA)

### Overview
NVIDIA's VLA for humanoid robots. 3B parameters, flow matching action transformer.

### Key Features
- Designed for humanoid platforms
- 20-40 demonstrations for fine-tuning
- Uses DreamGen for synthetic data
- Cross-embodiment via embodiment ID

### Architecture
- **Vision**: SigLip2 transformer
- **Language**: T5 transformer
- **Action**: Flow matching transformer

### Resources
- Research: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- HuggingFace: https://huggingface.co/nvidia/GR00T-N1.5-3B
- GitHub: https://github.com/NVIDIA/Isaac-GR00T
- Paper: https://arxiv.org/abs/2503.14734

---

## OpenVLA

### Overview
7B parameter open-source VLA trained on 970k real-world demonstrations.

### Key Features
- Built on Llama 2 7B
- Trained on Open X-Embodiment dataset
- Supports LoRA fine-tuning
- Outperforms RT-2-X (55B) by 16.5%

### Requirements
- 16GB+ GPU VRAM
- HuggingFace: `openvla/openvla-7b`

### Resources
- Project: https://openvla.github.io/
- GitHub: https://github.com/openvla/openvla
- Paper: https://arxiv.org/abs/2406.09246

---

## Model Selection Guide

### For Hackathon (Quick Results)

**Use ACT if:**
- First time with imitation learning
- Need fast training iteration
- Single-task manipulation
- Limited GPU resources

**Use SmolVLA if:**
- Need language conditioning ("pick up the red chip")
- Limited hardware (can run on CPU)
- Want pretrained model benefits

### For Complex Tasks

**Use Diffusion Policy if:**
- Complex manipulation with contacts
- Multi-modal action distributions
- Have GPU and time for training

### For Generalization

**Use Pi0/OpenVLA if:**
- Need multi-task capability
- Cross-robot transfer
- Have substantial GPU resources

---

## Data Collection Tips

### General Guidelines
- **50 demonstrations** is a good starting point for ACT
- **Consistency matters**: Same strategy across episodes
- **Clean data**: No failed demonstrations
- **Camera visibility**: Objects always in view

### For Language-Conditioned Models
- Match task description to actions
- Vary descriptions for same task (optional)
- Keep language simple and clear

### Demonstration Quality
1. Plan the task before recording
2. Move smoothly, avoid jerky motions
3. Complete the task fully
4. Keep workspace visible to cameras
5. Maintain consistent approach

---

## Training Tips

### ACT
- Start with default hyperparameters
- Train 100k steps initially
- Reduce batch size if OOM
- Learning rate: 1e-5

### SmolVLA
- Use pretrained checkpoint
- Fine-tune on your data
- Fewer demonstrations needed

### General
- Monitor with Weights & Biases
- Save checkpoints frequently
- Test early with 10 demos before full recording
