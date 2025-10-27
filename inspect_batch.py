import torch
import fire
import os
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DualStreamCollator:
    def __init__(self, tokenizer, max_length=512, objective='separate'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.objective = objective
        
    def __call__(self, batch):
        texts_A = [item['text'] for item in batch[::2]]
        texts_B = [item['text'] for item in batch[1::2]]
        
        min_len = min(len(texts_A), len(texts_B))
        texts_A = texts_A[:min_len]
        texts_B = texts_B[:min_len]
        
        encoded_A = self.tokenizer(
            texts_A,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoded_B = self.tokenizer(
            texts_B,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.objective == 'separate':
            labels_A = encoded_A['input_ids'].clone()
            labels_B = encoded_B['input_ids'].clone()
            labels_A[:, :-1] = encoded_A['input_ids'][:, 1:]
            labels_B[:, :-1] = encoded_B['input_ids'][:, 1:]
            labels_A[:, -1] = -100
            labels_B[:, -1] = -100
            labels_A[labels_A == self.tokenizer.pad_token_id] = -100
            labels_B[labels_B == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids_A': encoded_A['input_ids'],
                'attention_mask_A': encoded_A['attention_mask'],
                'labels_A': labels_A,
                'input_ids_B': encoded_B['input_ids'],
                'attention_mask_B': encoded_B['attention_mask'],
                'labels_B': labels_B,
            }
        elif self.objective == 'averaged':
            next_tokens_A = encoded_A['input_ids'][:, 1:].clone()
            next_tokens_B = encoded_B['input_ids'][:, 1:].clone()
            
            return {
                'input_ids_A': encoded_A['input_ids'],
                'attention_mask_A': encoded_A['attention_mask'],
                'next_tokens_A': next_tokens_A,
                'input_ids_B': encoded_B['input_ids'],
                'attention_mask_B': encoded_B['attention_mask'],
                'next_tokens_B': next_tokens_B,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


class AnomalyCheckingCollator(DualStreamCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_text_length = 10  # Min number of characters for a text to be valid

    def __call__(self, batch):
        # Perform anomaly checks on raw text before tokenization
        replacement_char = '\ufffd'
        for i, item in enumerate(batch):
            text = item.get('text', '')
            if not text or len(text.strip()) < self.min_text_length:
                print(f"\n[!] ANOMALY DETECTED: Item {i} in this batch has very short or empty text.")
                print(f"    Content: '{text[:100]}'")
            
            if replacement_char in text:
                print(f"\n[!] ANOMALY DETECTED: Item {i} contains Unicode replacement characters ({replacement_char}), indicating potential encoding issues.")
                char_index = text.find(replacement_char)
                start = max(0, char_index - 20)
                end = char_index + 20
                print(f"    Content snippet: '{text[start:end]}'")

        # Let the parent class handle the actual collation
        return super().__call__(batch)


def inspect_data(
    max_steps_to_check: int = 41000,
    model_name: str = 'HuggingFaceTB/SmolLM2-135M',
    dataset_name: str = 'roneneldan/TinyStories',
    dataset_config: str = None,
    dataset_split: str = 'train',
    batch_size: int = 8,
    max_length: int = 2048,
    seed: int = 42,
    objective: str = 'averaged',
    num_workers: int = 4
):
    """
    Inspects dataset for anomalies by simulating the training data loading process.

    Args:
        max_steps_to_check: How many data loader steps to check for anomalies.
        model_name: Name of the model to get the tokenizer from.
        dataset_name: Dataset name from HuggingFace.
        dataset_config: Dataset configuration (e.g., 'en' for C4).
        dataset_split: Dataset split to use.
        batch_size: Batch size per GPU (as used in training).
        max_length: Maximum sequence length.
        seed: Random seed for reproducibility.
        objective: The training objective ('separate' or 'averaged').
        num_workers: Number of worker processes for data loading.
    """
    print(f"--- Data Anomaly Inspection Utility ---")
    print(f"Checking up to {max_steps_to_check} steps with batch size {batch_size}...")
    print(f"Using {num_workers} workers for data loading.")

    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading dataset: {dataset_name} ({dataset_config or 'default'})")
    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True, trust_remote_code=True)
        
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    collate_fn = AnomalyCheckingCollator(tokenizer, max_length=max_length, objective=objective)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    data_iter = iter(dataloader)
    
    print(f"Starting inspection...")
    for current_step in tqdm(range(max_steps_to_check), desc="Inspecting batches"):
        try:
            batch = next(data_iter)
            
            # Post-collation checks
            padding_ratio_A = 1.0 - (batch['attention_mask_A'].sum() / batch['attention_mask_A'].numel())
            padding_ratio_B = 1.0 - (batch['attention_mask_B'].sum() / batch['attention_mask_B'].numel())

            if padding_ratio_A > 0.95 or padding_ratio_B > 0.95:
                print(f"\n[!] ANOMALY DETECTED (Step {current_step + 1}): High padding ratio detected.")
                print(f"    Padding A: {padding_ratio_A:.2%}, Padding B: {padding_ratio_B:.2%}")
                # Optional: print the content if highly padded
                if padding_ratio_A > 0.95:
                    print("    Highly padded content in A:")
                    for i in range(batch['input_ids_A'].size(0)):
                         if (1.0 - (batch['attention_mask_A'][i].sum() / batch['attention_mask_A'][i].numel())) > 0.95:
                            print(f"      - {tokenizer.decode(batch['input_ids_A'][i], skip_special_tokens=True)}")

        except StopIteration:
            print(f"\nDataset exhausted at step {current_step}. Inspection finished early.")
            break
        except Exception as e:
            print(f"\nAn error occurred at step {current_step}: {e}")
            continue

    print(f"\n{'='*80}")
    print("Inspection complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    fire.Fire(inspect_data)
