#!/usr/bin/env python3
"""
Copy TensorBoard events from one run to another up to a specific step.
"""
import os
import argparse
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm


def copy_tensorboard_events(source_dir, target_dir, max_step):
    """Copy TensorBoard events from source to target up to max_step."""
    
    print(f"Loading events from {source_dir}...")
    ea = event_accumulator.EventAccumulator(str(source_dir))
    ea.Reload()
    
    print(f"Creating new writer in {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=target_dir)
    
    # Get all available tags
    scalar_tags = ea.Tags().get('scalars', [])
    print(f"\nFound {len(scalar_tags)} scalar tags")
    
    # Copy scalars
    total_events = 0
    for tag in tqdm(scalar_tags, desc="Copying scalar tags"):
        events = ea.Scalars(tag)
        copied = 0
        for event in events:
            if event.step <= max_step:
                writer.add_scalar(tag, event.value, event.step, walltime=event.wall_time)
                copied += 1
        total_events += copied
    
    # Copy text events
    text_tags = ea.Tags().get('tensors', [])
    if text_tags:
        print(f"\nFound {len(text_tags)} text/tensor tags")
        for tag in tqdm(text_tags, desc="Copying text tags"):
            try:
                # Text events stored as tensors
                events = ea.Tensors(tag)
                for event in events:
                    if event.step <= max_step:
                        # For text, we need to decode the tensor value
                        try:
                            text_value = event.tensor_proto.string_val[0].decode('utf-8')
                            writer.add_text(tag, text_value, event.step, walltime=event.wall_time)
                        except:
                            pass  # Skip if can't decode
            except:
                pass  # Skip if can't load this tag
    
    writer.flush()
    writer.close()
    
    print(f"\n✅ Successfully copied {total_events} events up to step {max_step}")
    print(f"Target directory: {target_dir}")
    
    return target_dir


def main():
    parser = argparse.ArgumentParser(description='Copy TensorBoard events up to a specific step')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source TensorBoard log directory')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='Target TensorBoard log directory')
    parser.add_argument('--max_step', type=int, required=True,
                        help='Maximum step to copy (inclusive)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        raise ValueError(f"Source directory not found: {args.source_dir}")
    
    copy_tensorboard_events(args.source_dir, args.target_dir, args.max_step)


if __name__ == '__main__':
    main()

