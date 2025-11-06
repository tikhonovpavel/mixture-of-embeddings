# Refactoring Notes: Modular Architecture

## Overview

Исходный файл `train_dual_stream.py` (1459 строк) был разбит на несколько специализированных модулей для улучшения читаемости, поддерживаемости и переиспользования кода.

## Структура файлов

### 1. `train_dual_stream.py` (479 строк) — точка входа
**Новая главная функция**, которая координирует весь процесс обучения.

**Содержит:**
- `train()` — основная функция с параметрами обучения
- Логика обработки `continue_run` и возобновления обучения
- Инициализация всех компонентов (модель, данные, тренер)
- Вспомогательные функции для создания конфигурации и вывода информации

**Линии кода:**
- Координация компонентов
- Обработка параметров командной строки (через `fire`)
- Setup/cleanup распределённого обучения

---

### 2. `checkpoint_manager.py` (77 строк) — управление чекпоинтами
**Отвечает за:** загрузку и сохранение чекпоинтов модели.

**Основные методы:**
```python
class CheckpointManager:
    - find_latest_checkpoint() → Tuple[str, int]
    - save_checkpoint(step, model_state_dict, ...) → None
    - load_checkpoint(checkpoint_path) → dict
```

**Преимущества:**
- Инкапсулирует всю логику работы с файлами чекпоинтов
- Легко переиспользовать в других скриптах
- Чистый интерфейс для работы с чекпоинтами

---

### 3. `config_manager.py` (119 строк) — управление конфигурацией
**Отвечает за:** загрузку, сохранение и применение "живой" конфигурации.

**Основные методы:**
```python
class ConfigManager:
    - load_config() → Optional[Dict]
    - save_config(config: Dict) → None
    - create_initial_config(...) → Dict
    - apply_config_changes(current_config, new_config, optimizer, ...) → Dict[str, tuple]
```

**Преимущества:**
- Логика обновления гиперпараметров на лету выделена в отдельный класс
- Можно менять конфигурацию без перезагрузки скрипта (для live config)
- Чистый интерфейс для работы с параметрами

---

### 4. `logger.py` (325 строк) — логирование в TensorBoard
**Отвечает за:** весь логирующий функционал в TensorBoard.

**Основные методы:**
```python
class TrainingLogger:
    - copy_tensorboard_events(source_dir, max_step) → None
    - log_training_config(config, script_path) → None
    - log_generation_samples(samples, step) → None
    - log_training_metrics(step, total_loss, ...) → None
    - log_similarity_metrics(step, emb_similarity, ...) → None
    - log_config_change(step, changes) → None
    - log_initial_hyperparams(config, alpha, beta, ...) → None
    - log_scheduler_loss(step, loss) → None
    - close() → None
```

**Преимущества:**
- Весь логирующий код в одном месте
- Легко добавить новые метрики логирования
- Не загромождает основной класс тренера
- Можно использовать отдельно для других экспериментов

---

### 5. `generation.py` (160 строк) — генерация текста
**Отвечает за:** текстовую генерацию из модели во время обучения.

**Основные функции:**
```python
def generate_samples(model, tokenizer, prompts, device, max_new_tokens, temperature, objective) → List[Dict]
```

**Логика:**
- Поддерживает три режима генерации: `separate`, `averaged`, `mixup`
- Использует различные стратегии в зависимости от objective
- Возвращает структурированные результаты для логирования

**Преимущества:**
- Генерация изолирована, может быть использована отдельно
- Легко добавить новые стратегии генерации
- Чистый интерфейс для вызова

---

### 6. `collator.py` (126 строк) — подготовка батчей
**Отвечает за:** создание батчей для двух потоков обучения.

**Основная логика:**
```python
class DualStreamCollator:
    def __call__(self, batch) → Dict
```

**Особенности:**
- Создаёт парные батчи из двух потоков текста
- Поддерживает три objective: `separate`, `averaged`, `mixup`
- Для `separate`: создаёт labels для обоих потоков
- Для `averaged`/`mixup`: создаёт next_tokens и lambdas

**Преимущества:**
- Логика подготовки данных отделена от тренера
- Легко модифицировать логику создания батчей
- Переиспользуемо в других экспериментах

---

### 7. `utils.py` (65 строк) — утилиты
**Отвечает за:** вспомогательные функции общего назначения.

**Содержит:**
```python
- set_seed(seed) → None
- setup_distributed(device_override) → Tuple[int, int, int]
- cleanup_distributed() → None
- compute_weight_similarity(weight_A, weight_B) → float
- compute_weight_distance(weight_A, weight_B) → float
```

**Преимущества:**
- Все утилиты в одном месте
- Легко переиспользовать в других скриптах
- Чистая организация кода

---

### 8. `trainer.py` (712 строк) — основной класс тренера
**Отвечает за:** координацию процесса обучения.

**Основная структура:**
```python
class Trainer:
    - __init__(...) → None
    - train() → None
    
    # Приватные методы
    - _setup_optimizer() → None
    - _setup_live_config() → Dict
    - _load_checkpoint(checkpoint_data) → None
    - _compute_loss_separate(batch) → Tuple[Tensor, Tensor]
    - _compute_loss_averaged(batch) → Tuple[Tensor, Tensor]
    - _compute_loss_mixup(batch) → Tuple[Tensor, Tensor]
    - _update_loss_balancing_coefficients(step) → Tuple[float, float]
    - _check_and_apply_config_changes(step) → None
    - _log_similarity_metrics(optimizer_step) → None
    - _generate_and_log_samples(optimizer_step) → None
    - ... и другие приватные методы
```

**Основной метод `train()`:**
- Инициализирует состояние обучения
- Быстро перемещает dataloader при возобновлении
- Логирует начальные примеры
- **Основной цикл обучения:**
  - Получает батч
  - Вычисляет потери
  - Делает шаг оптимизатора
  - Логирует метрики
  - Сохраняет чекпоинты

**Преимущества:**
- Вся логика обучения в одном месте
- Методы для разных objective изолированы
- Легко добавить новые функции (новые objective, новые метрики)
- Тестируемо — каждый метод независим

---

## Сравнение размеров

| Компонент | Старый | Новый | Описание |
|-----------|--------|-------|---------|
| train_dual_stream.py | 1459 | 479 | Основной скрипт (уменьшен в 3 раза) |
| Всего новых модулей | - | 1064 | 8 новых файлов (но логичнее организованы) |
| **Итого кода** | **1459** | **1543** | +84 строк (из-за импортов и классов) |

## Преимущества рефакторинга

### 1. **Читаемость**
- Каждый модуль отвечает за одну задачу (принцип SRP)
- Названия функций и классов отражают их назначение
- Легче понять логику каждой части

### 2. **Поддерживаемость**
- Изменения в одном модуле не влияют на другие
- Проще найти и исправить баги
- Легче добавить новые функции

### 3. **Переиспользование кода**
- Каждый модуль можно импортировать отдельно
- Можно использовать компоненты в других скриптах
- Легче писать тесты

### 4. **Масштабируемость**
- Легко добавить новые objective
- Легко добавить новые метрики логирования
- Можно расширять функциональность без изменения существующего кода

## Как использовать

### Обучение с новым скриптом

```bash
 python train_dual_stream.py \
    --model_name='HuggingFaceTB/SmolLM2-135M' \
    --dataset_name='roneneldan/TinyStories' \
    --num_train_steps=200000 \
    --objective='separate' \
    --use_loss_balancing=True
```

### Использование отдельных модулей

```python
# Использование CheckpointManager в другом скрипте
from checkpoint_manager import CheckpointManager

manager = CheckpointManager('./checkpoints')
latest = manager.find_latest_checkpoint()
if latest:
    path, step = latest
    checkpoint = manager.load_checkpoint(path)
    print(f"Loaded checkpoint from step {step}")

# Использование ConfigManager
from config_manager import ConfigManager

config_mgr = ConfigManager('./config.json')
config = config_mgr.load_config()
config['learning_rate'] = 1e-4
config_mgr.save_config(config)

# Использование утилит
from utils import set_seed, compute_weight_similarity
import torch

set_seed(42)
weight_A = torch.randn(100, 100)
weight_B = torch.randn(100, 100)
similarity = compute_weight_similarity(weight_A, weight_B)
print(f"Similarity: {similarity}")
```

## Структура директории

```
mixture-of-embeddings/
├── train_dual_stream.py          # Точка входа (479 строк)
├── train_dual_stream_old.py      # Старая версия (резервная копия)
├── trainer.py                    # Основной класс Trainer (712 строк)
├── checkpoint_manager.py         # Управление чекпоинтами (77 строк)
├── config_manager.py             # Управление конфигурацией (119 строк)
├── logger.py                     # TensorBoard логирование (325 строк)
├── generation.py                 # Текстовая генерация (160 строк)
├── collator.py                   # Подготовка батчей (126 строк)
├── utils.py                      # Утилиты (65 строк)
├── model.py                      # (существующий файл)
├── REFACTORING_NOTES.md          # Этот файл
└── ...
```

