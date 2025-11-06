# Примеры использования модулей

## 1. Базовое обучение

### Простой запуск с параметрами по умолчанию
```bash
 python train_dual_stream.py
```

### С пользовательскими параметрами
```bash
 python train_dual_stream.py \
    --model_name='HuggingFaceTB/SmolLM2-135M' \
    --dataset_name='roneneldan/TinyStories' \
    --batch_size=16 \
    --learning_rate=1e-3 \
    --num_train_steps=100000 \
    --objective='separate' \
    --use_loss_balancing=True \
    --output_dir='./my_checkpoints' \
    --tensorboard_log_dir='./my_logs'
```

### Обучение с different objective
```bash
# Averaged objective
 python train_dual_stream.py --objective='averaged'

# Mixup objective (требует shared_embeddings=True)
 python train_dual_stream.py --objective='mixup' --shared_embeddings=True
```

---

## 2. Возобновление обучения

### Продолжить последний запуск с новой директорией
```bash
 python train_dual_stream.py \
    --continue_run='./checkpoints/run_20241106_120000' \
    --resume_from_checkpoint='auto'
```

### Возобновить с конкретного чекпоинта
```bash
 python train_dual_stream.py \
    --continue_run='./checkpoints/run_20241106_120000' \
    --resume_from_checkpoint='./checkpoints/run_20241106_120000/checkpoint-50000/model.pt'
```

---

## 3. Использование отдельных модулей

### CheckpointManager — работа с чекпоинтами

```python
import torch
from checkpoint_manager import CheckpointManager

# Инициализация
checkpoint_mgr = CheckpointManager('./my_checkpoints')

# Найти последний чекпоинт
result = checkpoint_mgr.find_latest_checkpoint()
if result:
    checkpoint_path, step = result
    print(f"Найден чекпоинт на шаге {step}")
    checkpoint = checkpoint_mgr.load_checkpoint(checkpoint_path)
    
    # Загрузить состояние модели
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print("Чекпоинтов не найдено")

# Сохранить чекпоинт
checkpoint_mgr.save_checkpoint(
    step=100,
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    scheduler_state_dict=scheduler.state_dict(),
    alpha=1.2,
    beta=0.8,
    ema_loss_A=2.5,
    ema_loss_B=2.3
)
```

### ConfigManager — управление конфигурацией

```python
from config_manager import ConfigManager

# Инициализация
config_mgr = ConfigManager('./config.json')

# Создать начальную конфигурацию
config = config_mgr.create_initial_config(
    learning_rate=1e-3,
    backbone_lr=1e-4,
    temperature=0.7,
    log_interval=10,
    similarity_log_interval=50,
    eval_interval=500,
    loss_balance_strength=5.0,
    loss_balance_ema_momentum=0.99,
    loss_balance_clip_min=0.1,
    loss_balance_clip_max=10.0
)

# Загрузить конфигурацию
config = config_mgr.load_config()
print(f"Текущий learning rate: {config['learning_rate']}")

# Обновить конфигурацию вручную
config['learning_rate'] = 5e-4
config_mgr.save_config(config)

# Применить изменения конфигурации к optimizer
changes = config_mgr.apply_config_changes(
    current_config=old_config,
    new_config=new_config,
    optimizer=optimizer,
    freeze_backbone=False
)

if changes:
    for param_name, (old_val, new_val) in changes.items():
        print(f"{param_name}: {old_val} -> {new_val}")
```

### TrainingLogger — логирование в TensorBoard

```python
from logger import TrainingLogger

# Инициализация (только на rank 0 будет создавать writer)
logger = TrainingLogger(log_dir='./tensorboard_logs', rank=0)

# Логировать конфигурацию
logger.log_training_config(
    config={
        'model_name': 'SmolLM2-135M',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'objective': 'separate'
    },
    script_path='./train_dual_stream.py'
)

# Логировать метрики обучения
logger.log_training_metrics(
    step=100,
    total_loss=2.5,
    loss_A=2.3,
    loss_B=2.7,
    ema_loss_A=2.4,
    ema_loss_B=2.6,
    alpha=1.1,
    beta=0.9,
    learning_rate=1e-3,
    backbone_lr=1e-4,
    grad_norm=1.5,
    objective='separate',
    use_loss_balancing=True
)

# Логировать метрики похожести эмбеддингов
logger.log_similarity_metrics(
    step=100,
    emb_similarity=0.95,
    emb_distance=0.05,
    unemb_similarity=0.92,
    unemb_distance=0.08
)

# Логировать сгенерированные примеры
samples = [
    {
        'prompt_A': 'The capital of',
        'prompt_B': 'Machine learning is',
        'generated_A': 'The capital of France is Paris.',
        'generated_B': 'Machine learning is a subset of AI.'
    }
]
logger.log_generation_samples(samples, step=100)

# Логировать изменения конфигурации
changes = {
    'learning_rate': (1e-3, 5e-4),
    'temperature': (0.7, 0.8)
}
logger.log_config_change(step=100, changes=changes)

# Закрыть writer
logger.close()
```

### generation.py — генерация текста

```python
from generation import generate_samples
import torch

# Модель и токенизатор должны быть готовы
model.eval()

prompts = [
    ("The capital of", "Machine learning is"),
    ("Once upon a", "In the field of"),
]

samples = generate_samples(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    device=device,
    max_new_tokens=50,
    temperature=0.7,
    objective='separate'  # или 'averaged', 'mixup'
)

# Результаты
for i, sample in enumerate(samples):
    print(f"\nSample {i+1}:")
    print(f"  Prompt A: {sample['prompt_A']}")
    print(f"  Generated A: {sample['generated_A']}")
    print(f"  Prompt B: {sample['prompt_B']}")
    print(f"  Generated B: {sample['generated_B']}")
```

### DualStreamCollator — подготовка батчей

```python
from collator import DualStreamCollator
from torch.utils.data import DataLoader
from datasets import load_dataset

# Загрузить датасет
dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

# Создать collator для objective='separate'
collator_separate = DualStreamCollator(
    tokenizer=tokenizer,
    max_length=512,
    objective='separate',
    mixup_alpha=1.0
)

# Создать collator для objective='mixup'
collator_mixup = DualStreamCollator(
    tokenizer=tokenizer,
    max_length=512,
    objective='mixup',
    mixup_alpha=0.4
)

# Создать dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator_separate
)

# Использовать в цикле обучения
for batch in dataloader:
    input_ids_A = batch['input_ids_A']
    attention_mask_A = batch['attention_mask_A']
    labels_A = batch['labels_A']  # только для 'separate'
    
    input_ids_B = batch['input_ids_B']
    attention_mask_B = batch['attention_mask_B']
    labels_B = batch['labels_B']  # только для 'separate'
    
    # Обучение...
    break  # Только для примера
```

### utils.py — утилиты

```python
from utils import set_seed, compute_weight_similarity, compute_weight_distance
import torch

# Установить seed
set_seed(42)

# Вычислить похожесть между двумя матрицами весов
weight_A = torch.randn(100, 100)
weight_B = torch.randn(100, 100)

similarity = compute_weight_similarity(weight_A, weight_B)
print(f"Cosine similarity: {similarity:.4f}")

distance = compute_weight_distance(weight_A, weight_B)
print(f"Frobenius distance: {distance:.4f}")

# Distributed setup
from utils import setup_distributed, cleanup_distributed

rank, world_size, local_rank = setup_distributed(device_override=None)
print(f"Rank: {rank}, World size: {world_size}")

# ... обучение ...

cleanup_distributed()
```

---

## 4. Расширенные примеры

### Полный цикл обучения с нуля

```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

from model import DualStreamLLM
from collator import DualStreamCollator
from trainer import Trainer
from utils import set_seed, setup_distributed

# Setup
set_seed(42)
rank, world_size, local_rank = setup_distributed(device=None)
device = torch.device(f'cuda:{local_rank}')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)

# Create model
config = AutoConfig.from_pretrained('HuggingFaceTB/SmolLM2-135M')
model = DualStreamLLM(
    'HuggingFaceTB/SmolLM2-135M',
    config.vocab_size,
    config.hidden_size,
    freeze_backbone=False,
    objective='separate',
    shared_embeddings=False
)
model = model.to(device)

# Create dataloader
collator = DualStreamCollator(tokenizer, max_length=512, objective='separate')
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

# Create trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    dataloader=dataloader,
    device=device,
    rank=rank,
    world_size=world_size,
    local_rank=local_rank,
    output_dir='./checkpoints',
    tensorboard_log_dir='./tensorboard_logs',
    learning_rate=1e-3,
    backbone_lr=1e-4,
    freeze_backbone=False,
    num_train_steps=100000,
    gradient_accumulation_steps=1,
    objective='separate',
    use_loss_balancing=True,
    shared_embeddings=False,
    loss_balance_strength=5.0,
    loss_balance_ema_momentum=0.99,
    loss_balance_warmup_steps=100,
    loss_balance_clip_min=0.1,
    loss_balance_clip_max=10.0,
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.5,
    log_interval=1,
    similarity_log_interval=50,
    eval_interval=500,
    save_interval=5000,
    max_gen_tokens=50,
    temperature=0.7,
    validation_prompts=[
        ("The capital of", "Machine learning is"),
        ("Once upon a", "In the field of"),
    ]
)

# Train
trainer.train()
```

### Использование в другом скрипте для вывода

```python
import torch
from transformers import AutoTokenizer
from generation import generate_samples
from checkpoint_manager import CheckpointManager

# Load model and tokenizer
model_name = 'HuggingFaceTB/SmolLM2-135M'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Find and load checkpoint
checkpoint_mgr = CheckpointManager('./checkpoints/run_20241106_120000')
result = checkpoint_mgr.find_latest_checkpoint()
if result:
    checkpoint_path, step = result
    checkpoint = checkpoint_mgr.load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Checkpoint not found!")
    exit(1)

model.eval()
device = torch.device('cuda:0')
model = model.to(device)

# Generate samples
prompts = [
    ("Once upon a time", "The future of AI"),
    ("Mathematics is", "Programming languages"),
]

samples = generate_samples(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    device=device,
    max_new_tokens=100,
    temperature=0.7,
    objective='separate'
)

for sample in samples:
    print(f"Prompt A: {sample['prompt_A']}")
    print(f"Generated A: {sample['generated_A']}\n")
    print(f"Prompt B: {sample['prompt_B']}")
    print(f"Generated B: {sample['generated_B']}\n")
    print("="*80)
```

---

## 5. Советы и tricks

### Мониторинг обучения через TensorBoard

```bash
 tensorboard --logdir=./tensorboard_logs
```

Затем откройте http://localhost:6006 в браузере.

### Динамическое изменение learning rate

1. Обучение запущено
2. Отредактируйте `./checkpoints/run_XXXXX/live_config.json`:
   ```json
   {
     "learning_rate": 5e-4,
     "backbone_lr": 5e-5,
     ...
   }
   ```
3. Скрипт проверит изменения каждые 10 шагов и применит их

### Оптимизация памяти

```bash
# Уменьшить batch size
 python train_dual_stream.py --batch_size=8

# Использовать gradient accumulation
 python train_dual_stream.py --batch_size=16 --gradient_accumulation_steps=4

# Включить FSDP для распределённого обучения
 torchrun --nproc_per_node=4 train_dual_stream.py --batch_size=32
```

### Отладка проблем

```python
# Если возникла ошибка non-finite loss
# Проверьте сохранённый батч:
import torch

batch = torch.load('./checkpoints/run_XXXXX/nonfinite_loss_batch_step_YYY.pt')
print(batch.keys())
print(batch['input_ids_A'].shape)
```

