import torch
import numpy as np


class DualStreamCollator:
    """Collator for dual-stream training that creates paired batches."""
    
    def __init__(self, tokenizer, max_length: int = 512, objective: str = 'separate', mixup_alpha: float = 1.0):
        """Initialize collator.
        
        Args:
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            objective: Training objective ('separate', 'averaged', or 'mixup')
            mixup_alpha: Alpha parameter for Beta distribution in mixup
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.objective = objective
        self.mixup_alpha = mixup_alpha
        
    def __call__(self, batch):
        """Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dictionary containing encoded inputs for both streams
        """
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
            # Shift labels: predict next token, not current
            labels_A = encoded_A['input_ids'].clone()
            labels_B = encoded_B['input_ids'].clone()
            
            # Shift left: labels[i] = input_ids[i+1]
            labels_A[:, :-1] = encoded_A['input_ids'][:, 1:]
            labels_B[:, :-1] = encoded_B['input_ids'][:, 1:]
            
            # Last position has no next token, so ignore it
            labels_A[:, -1] = -100
            labels_B[:, -1] = -100
            
            # Also ignore padding tokens
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
            # For averaged objective, we need to create target distributions
            # Get next token ids
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
        
        elif self.objective == 'mixup':
            # Sample mixing coefficients from Beta distribution for each sample in batch
            batch_size = encoded_A['input_ids'].size(0)
            # Sample λ ~ Beta(α, α) for each sample
            lambdas = torch.from_numpy(
                np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            ).float()
            
            # Get next token ids for creating soft targets
            next_tokens_A = encoded_A['input_ids'][:, 1:].clone()
            next_tokens_B = encoded_B['input_ids'][:, 1:].clone()
            
            return {
                'input_ids_A': encoded_A['input_ids'],
                'attention_mask_A': encoded_A['attention_mask'],
                'next_tokens_A': next_tokens_A,
                'input_ids_B': encoded_B['input_ids'],
                'attention_mask_B': encoded_B['attention_mask'],
                'next_tokens_B': next_tokens_B,
                'lambdas': lambdas,  # [batch_size]
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

