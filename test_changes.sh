#!/bin/bash

# Test with default parameters (TinyStories + SmolLM2-135M + separate objective)
echo "Testing separate objective..."
 python train_dual_stream.py \
    --num_train_steps=5 \
    --batch_size=2 \
    --max_length=128 \
    --log_interval=1 \
    --save_interval=1000 \
    --eval_interval=1000 \
    --similarity_log_interval=1000 \
    --tensorboard_log_dir="" \
    --device="cuda:0"

echo ""
echo "Testing averaged objective..."
 python train_dual_stream.py \
    --objective="averaged" \
    --num_train_steps=5 \
    --batch_size=2 \
    --max_length=128 \
    --log_interval=1 \
    --save_interval=1000 \
    --eval_interval=1000 \
    --similarity_log_interval=1000 \
    --tensorboard_log_dir="" \
    --device="cuda:0"

echo ""
echo "Testing with C4 dataset..."
 python train_dual_stream.py \
    --dataset_name="allenai/c4" \
    --dataset_config="en" \
    --num_train_steps=5 \
    --batch_size=2 \
    --max_length=128 \
    --log_interval=1 \
    --save_interval=1000 \
    --eval_interval=1000 \
    --similarity_log_interval=1000 \
    --tensorboard_log_dir="" \
    --device="cuda:0"

echo ""
echo "Testing without loss balancing..."
 python train_dual_stream.py \
    --use_loss_balancing=False \
    --num_train_steps=5 \
    --batch_size=2 \
    --max_length=128 \
    --log_interval=1 \
    --save_interval=1000 \
    --eval_interval=1000 \
    --similarity_log_interval=1000 \
    --tensorboard_log_dir="" \
    --device="cuda:0"

echo ""
echo "All tests completed!"

