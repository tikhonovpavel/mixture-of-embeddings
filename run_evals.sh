#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH="/disk/4tb/tikhonov/projects/mixture-of-embeddings"
RESULTS_DIR="${BASE_PATH}/all_runs_analysis"
SEPARATOR="EleutherAI/pythia-160m"

# ЕДИНЫЙ КЭШ ДЛЯ ВСЕХ МОДЕЛЕЙ (FineWeb)
# Скипаем с запасом: 250000 * 8 * 2 = 4M сэмплов. 
# Это больше, чем видела любая из моделей.
COMMON_CACHE="${RESULTS_DIR}/common_eval_cache_fineweb_deep.json"

mkdir -p "${RESULTS_DIR}"

echo "========================================================"
echo "STARTING STANDARDIZED EVALUATION"
echo "Results dir: ${RESULTS_DIR}"
echo "Common Cache: ${COMMON_CACHE}"
echo "========================================================"

# Функция для запуска
run_eval() {
    local run_name=$1
    local step=$2
    local bs=$3
    local sep_bool=$4
    local desc=$5
    
    local checkpoint="${BASE_PATH}/checkpoints/${run_name}/checkpoint-${step}/model.pt"
    local config="${BASE_PATH}/checkpoints/${run_name}/training_config.json"
    local json_out="${RESULTS_DIR}/${run_name}_step${step}.json"
    local txt_out="${RESULTS_DIR}/${run_name}_step${step}.txt"

    if [ ! -f "$checkpoint" ]; then
        echo "❌ SKIP: Checkpoint not found: $checkpoint"
        return
    fi

    echo "▶ Evaluating: ${desc}"
    
    # ПРИМЕЧАНИЕ:
    # num_train_steps и train_batch_size здесь важны ТОЛЬКО для первого запуска,
    # который создаст кэш. Для остальных они будут проигнорированы, так как
    # скрипт увидит существующий cached_dataset_path.
    # Мы ставим большие значения (250000 / 8), чтобы гарантировать hold-out для всех.
    
    python evaluate_dual_stream.py evaluate \
        --checkpoint_path "${checkpoint}" \
        --config_path "${config}" \
        --dataset_name "HuggingFaceFW/fineweb" \
        --num_samples 200 \
        --context_len 32 --eval_len 16 \
        --num_train_steps 250000 --train_batch_size 8 \
        --seed 42 \
        --use_separator_model "${sep_bool}" --separator_model_name "${SEPARATOR}" \
        --cached_dataset_path "${COMMON_CACHE}" \
        --output_json_path "${json_out}" \
        > "${txt_out}" 2>&1
        
    echo "  ✅ Done. Result: $(basename "$json_out")"
}

# # ==========================================
# # 1. Pythia 2.8B (FineWeb)
# # Separator=True
# # ==========================================

# # Run 4: Distillation New (Используем его первым для генерации кэша)
# run_eval "run_20251202_085040" 200000 8 "True" "Pythia 2.8B Distill New (200k)"

# # Run 3: Distillation Old (40k)
# run_eval "run_20251107_021028" 40000 8 "True" "Pythia 2.8B Distill Old (40k)"

# # Run 6: Separate GT (200k)
# run_eval "run_20251224_185822" 200000 8 "True" "Pythia 2.8B Separate GT (200k)"

# # Run 7: Separate Distillation (160k)
# run_eval "run_20251228_230459" 160000 8 "True" "Pythia 2.8B Sep Distill (160k)"


# # ==========================================
# # 2. Large Models (FineWeb)
# # Используют тот же кэш, так что сравнение будет честным
# # ==========================================

# # Run 8: Pythia 6.9B (400k)
# # Separator=True
# run_eval "run_20260102_155906" 400000 2 "True" "Pythia 6.9B Sep Distill (400k)"

# Run 9: Llama 3.1 8B (240k)
# Separator=False (Tokenizer mismatch)
run_eval "run_20260107_110217" 160000 2 "False" "Llama 3.1 Sep Distill (160k)"

echo "All evaluations finished!"