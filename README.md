# Modular Architecture

## Overview

### 1. `train_dual_stream.py` — Entry Point

**The main function** that coordinates the entire training process.

**Contains:**
- `train()` — The primary function with training parameters.
- Logic for handling `continue_run` and resuming training.
- Initialization of all components (model, data, trainer).
- Helper functions for creating the configuration and printing information.

**Lines of code handle:**
- Coordination of components.
- Processing of command-line arguments (via `fire`).
- Setup/cleanup of distributed training.

---

### 2. `checkpoint_manager.py` — Checkpoint Management
**Responsible for:** Loading and saving model checkpoints.

**Main methods:**
```python
class CheckpointManager:
    - find_latest_checkpoint() -> Tuple[str, int]
    - save_checkpoint(step, model_state_dict, ...) -> None
    - load_checkpoint(checkpoint_path) -> dict
```

**Advantages:**
- Encapsulates all logic for working with checkpoint files.
- Easy to reuse in other scripts.
- Clean interface for working with checkpoints.

---

### 3. `config_manager.py` — Configuration Management
**Responsible for:** Loading, saving, and applying the "live" configuration.

**Main methods:**
```python
class ConfigManager:
    - load_config() -> Optional[Dict]
    - save_config(config: Dict) -> None
    - create_initial_config(...) -> Dict
    - apply_config_changes(current_config, new_config, optimizer, ...) -> Dict[str, tuple]
```

**Advantages:**
- Logic for updating hyperparameters on the fly is isolated in a separate class.
- Configuration can be changed without restarting the script (for live config).
- Clean interface for working with parameters.

---

### 4. `logger.py` — TensorBoard Logging
**Responsible for:** All logging functionality to TensorBoard.

**Main methods:**
```python
class TrainingLogger:
    - copy_tensorboard_events(source_dir, max_step) -> None
    - log_training_config(config, script_path) -> None
    - log_generation_samples(samples, step) -> None
    - log_training_metrics(step, total_loss, ...) -> None
    - log_similarity_metrics(step, emb_similarity, ...) -> None
    - log_config_change(step, changes) -> None
    - log_initial_hyperparams(config, alpha, beta, ...) -> None
    - log_scheduler_loss(step, loss) -> None
    - close() -> None
```

**Advantages:**
- All logging code is in one place.
- Easy to add new logging metrics.
- Does not clutter the main trainer class.
- Can be used separately for other experiments.

---

### 5. `generation.py` — Text Generation
**Responsible for:** Generating text from the model during training.

**Main functions:**
```python
def generate_samples(model, tokenizer, prompts, device, max_new_tokens, temperature, objective) -> List[Dict]
```

**Logic:**
- Supports three generation modes: `separate`, `averaged`, `mixup`.
- Uses different strategies depending on the objective.
- Returns structured results for logging.

**Advantages:**
- Generation logic is isolated and can be used separately.
- Easy to add new generation strategies.
- Clean interface for invocation.

---

### 6. `collator.py` — Batch Preparation
**Responsible for:** Creating batches for the two training streams.

**Main logic:**
```python
class DualStreamCollator:
    def __call__(self, batch) -> Dict
```

**Features:**
- Creates paired batches from two text streams.
- Supports three objectives: `separate`, `averaged`, `mixup`.
- For `separate`: creates labels for both streams.
- For `averaged`/`mixup`: creates `next_tokens` and `lambdas`.

**Advantages:**
- Data preparation logic is separated from the trainer.
- Easy to modify the batch creation logic.
- Reusable in other experiments.

---

### 7. `utils.py` — Utilities
**Responsible for:** General-purpose helper functions.

**Contains:**
```python
- set_seed(seed) -> None
- setup_distributed(device_override) -> Tuple[int, int, int]
- cleanup_distributed() -> None
- compute_weight_similarity(weight_A, weight_B) -> float
- compute_weight_distance(weight_A, weight_B) -> float
```

**Advantages:**
- All utilities are in one place.
- Easy to reuse in other scripts.
- Clean code organization.

---

### 8. `trainer.py` — Main Trainer Class
**Responsible for:** Coordinating the training process.

**Main structure:**
```python
class Trainer:
    - __init__(...) -> None
    - train() -> None
    
    # Private methods
    - _setup_optimizer() -> None
    - _setup_live_config() -> Dict
    - _load_checkpoint(checkpoint_data) -> None
    - _compute_loss_separate(batch) -> Tuple[Tensor, Tensor]
    - _compute_loss_averaged(batch) -> Tuple[Tensor, Tensor]
    - _compute_loss_mixup(batch) -> Tuple[Tensor, Tensor]
    - _update_loss_balancing_coefficients(step) -> Tuple[float, float]
    - _check_and_apply_config_changes(step) -> None
    - _log_similarity_metrics(optimizer_step) -> None
    - _generate_and_log_samples(optimizer_step) -> None
    - ... and other private methods
```

**Main `train()` method:**
- Initializes the training state.
- Quickly seeks the dataloader position when resuming.
- Logs initial samples.
- **Main training loop:**
  - Gets a batch.
  - Computes losses.
  - Takes an optimizer step.
  - Logs metrics.
  - Saves checkpoints.

---

## How to Use

### Training with the New Script

```bash
 python train_dual_stream.py \
    --model_name='HuggingFaceTB/SmolLM2-135M' \
    --dataset_name='roneneldan/TinyStories' \
    --num_train_steps=200000 \
    --objective='separate' \
    --use_loss_balancing=True
```

### Training with Distillation

The distillation objective uses a teacher model to create soft targets by averaging the probability distributions from both text streams:

```bash
 python train_dual_stream.py \
    --model_name='HuggingFaceTB/SmolLM2-135M' \
    --dataset_name='roneneldan/TinyStories' \
    --num_train_steps=200000 \
    --objective='distillation' \
    --teacher_model_name='HuggingFaceTB/SmolLM2-135M' \
    --teacher_load_in_8bit=True \
    --distillation_temperature=2.0
```

**How it works:**
1. For each batch, the teacher model processes both text streams A and B separately (without mixing embeddings)
2. The teacher produces soft probability distributions P_teacher_A and P_teacher_B using temperature scaling
3. The target distribution is computed as: P_target = 0.5 * P_teacher_A + 0.5 * P_teacher_B
4. The student model (with mixed embeddings) is trained using KL divergence between its predictions and P_target

**Parameters:**
- `teacher_model_name`: Name of the teacher model (defaults to same as student model)
- `teacher_load_in_8bit`: Load teacher in 8-bit quantization mode to save memory (default: True)
- `distillation_temperature`: Temperature for softening distributions (default: 2.0)

### Using Individual Modules

```python
# Using CheckpointManager in another script
from checkpoint_manager import CheckpointManager

manager = CheckpointManager('./checkpoints')
latest = manager.find_latest_checkpoint()
if latest:
    path, step = latest
    checkpoint = manager.load_checkpoint(path)
    print(f"Loaded checkpoint from step {step}")

# Using ConfigManager
from config_manager import ConfigManager

config_mgr = ConfigManager('./config.json')
config = config_mgr.load_config()
config['learning_rate'] = 1e-4
config_mgr.save_config(config)

# Using utilities
from utils import set_seed, compute_weight_similarity
import torch

set_seed(42)
weight_A = torch.randn(100, 100)
weight_B = torch.randn(100, 100)
similarity = compute_weight_similarity(weight_A, weight_B)
print(f"Similarity: {similarity}")
```

## Directory Structure

```
mixture-of-embeddings/
├── train_dual_stream.py          # Entry point
├── trainer.py                    # Main Trainer class
├── checkpoint_manager.py         # Checkpoint management
├── config_manager.py             # Configuration management
├── logger.py                     # TensorBoard logging
├── generation.py                 # Text generation
├── collator.py                   # Batch preparation
├── utils.py                      # Utilities
└── model.py                      # Model
```