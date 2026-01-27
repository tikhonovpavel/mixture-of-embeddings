import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig

class DualStreamLLM(nn.Module):
    def __init__(self, model_name, freeze_backbone=True,
                 freeze_embeddings=False, optimize_memory=False,
                 embedding_init_strategy='random', embedding_init_noise_std=0.01,
                 unembedding_init_strategy=None,
                 objective='separate', shared_embeddings=False,
                 use_device_map_auto=False):
        super().__init__()
        self.objective = objective
        self.shared_embeddings = shared_embeddings
        
        if unembedding_init_strategy is None:
            unembedding_init_strategy = embedding_init_strategy
        
        # Load backbone with optional device_map='auto' for multi-GPU support
        backbone_kwargs = {
            'dtype': torch.bfloat16,
            'use_cache': False
        }
        if use_device_map_auto:
            backbone_kwargs['device_map'] = 'auto'
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            **backbone_kwargs
        )
        
        # Store whether we used device_map for later reference
        self.use_device_map_auto = use_device_map_auto
        
        base_model_config = AutoConfig.from_pretrained(model_name)
        vocab_size = base_model_config.vocab_size
        hidden_dim = base_model_config.hidden_size
        
        # Determine device for embeddings/unembeddings
        # If using device_map='auto', get device from backbone's first parameter
        if use_device_map_auto:
            embed_device = next(self.backbone.parameters()).device
        else:
            embed_device = torch.device('cpu')  # Will be moved to correct device later with .to()
        
        # Embedding initialization
        if shared_embeddings:
            if optimize_memory:
                # Use the backbone's embedding matrix directly to save memory
                self.emb = self.backbone.get_input_embeddings()
            else:
                # Create a new shared embedding matrix for both streams
                self.emb = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16, device=embed_device)
        else:
            # Separate embedding matrices
            self.emb_A = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16, device=embed_device)
            self.emb_B = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16, device=embed_device)
        
        # Unembedding initialization
        if objective == 'separate' or objective == 'separate_distillation':
            self.unemb_A = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16, device=embed_device)
            self.unemb_B = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16, device=embed_device)
        elif objective == 'averaged' or objective == 'mixup' or objective == 'distillation':
            self.unemb = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16, device=embed_device)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        self.stream_proj_A = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16, device=embed_device)
        self.stream_proj_B = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16, device=embed_device)
        
        # КРИТИЧНО ВАЖНО: Инициализация Identity + шум
        # Чтобы на старте Input = A + B
        with torch.no_grad():
            self.stream_proj_A.weight.copy_(torch.eye(hidden_dim))
            self.stream_proj_B.weight.copy_(torch.eye(hidden_dim))
            
            # Добавляем крошечный шум, чтобы сломать симметрию с первого шага
            self.stream_proj_A.weight.add_(torch.randn_like(self.stream_proj_A.weight) * 0.001)
            self.stream_proj_B.weight.add_(torch.randn_like(self.stream_proj_B.weight) * 0.001)
        
        # Initialize weights
        
        def _initialize_weights(target_embedding, base_weights, strategy, noise_std):
            if strategy == 'random':
                nn.init.normal_(target_embedding.weight, mean=0.0, std=0.02)
            elif strategy == 'copy':
                target_embedding.weight.data.copy_(base_weights)
            elif strategy == 'noisy_copy':
                noise = torch.randn_like(base_weights) * noise_std
                target_embedding.weight.data.copy_(base_weights + noise)
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")

        original_embed_weights = self.backbone.get_input_embeddings().weight.data.clone()
        if hasattr(self.backbone, 'get_output_embeddings') and self.backbone.get_output_embeddings() is not None:
            original_lm_head_weights = self.backbone.get_output_embeddings().weight.data.clone()
        else:
            original_lm_head_weights = original_embed_weights

        # Initialize embeddings
        # Note: If optimize_memory is True for shared_embeddings, self.emb IS the backbone embedding.
        # Initializing it will modify the backbone. This is generally expected behavior unless frozen.
        if shared_embeddings:
            _initialize_weights(self.emb, original_embed_weights, embedding_init_strategy, embedding_init_noise_std)
        else:
            _initialize_weights(self.emb_A, original_embed_weights, embedding_init_strategy, embedding_init_noise_std)
            # For B, we add noise only if the strategy is 'noisy_copy'. Otherwise, it's a direct copy or random.
            if embedding_init_strategy == 'noisy_copy':
                 _initialize_weights(self.emb_B, original_embed_weights, 'noisy_copy', embedding_init_noise_std)
            else:
                 _initialize_weights(self.emb_B, original_embed_weights, embedding_init_strategy, embedding_init_noise_std)
        
        # Initialize unembeddings
        if objective == 'separate' or objective == 'separate_distillation':
            _initialize_weights(self.unemb_A, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)
            # For B, we add noise only if the strategy is 'noisy_copy'.
            if unembedding_init_strategy == 'noisy_copy':
                _initialize_weights(self.unemb_B, original_lm_head_weights, 'noisy_copy', embedding_init_noise_std)
            else:
                _initialize_weights(self.unemb_B, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)
        else: # 'averaged' or 'mixup' objective
            _initialize_weights(self.unemb, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)

        # Freeze/Optimize Memory Cleanup
        
        # Apply explicit embedding freeze if requested
        if freeze_embeddings:
            if shared_embeddings:
                self.emb.weight.requires_grad = False
            else:
                self.emb_A.weight.requires_grad = False
                self.emb_B.weight.requires_grad = False
        
        # Optimize memory by removing unused backbone head
        if optimize_memory:
            if hasattr(self.backbone, 'lm_head'):
                self.backbone.lm_head = None
            elif hasattr(self.backbone, 'embed_out'):
                self.backbone.embed_out = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if hasattr(self.backbone, 'model'):
            self._transformer_attr = 'model'
        elif hasattr(self.backbone, 'transformer'):
            self._transformer_attr = 'transformer'
        elif hasattr(self.backbone, 'gpt_neox'):
            self._transformer_attr = 'gpt_neox'
        else:
            raise AttributeError("The backbone model does not have a standard transformer attribute ('model', 'transformer', 'gpt_neox').")
        
    @property
    def transformer(self):
        return getattr(self.backbone, self._transformer_attr)

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, lambdas=None):
        # Get embeddings
        if self.shared_embeddings:
            # Use same embedding matrix for both streams
            # Handle device mismatch when using device_map='auto'
            if self.use_device_map_auto:
                input_ids_A = input_ids_A.to(self.emb.weight.device)
                input_ids_B = input_ids_B.to(self.emb.weight.device)
                attention_mask_A = attention_mask_A.to(self.emb.weight.device)
                attention_mask_B = attention_mask_B.to(self.emb.weight.device)
            embeds_A = self.emb(input_ids_A)
            embeds_B = self.emb(input_ids_B)
        else:
            # Use separate embedding matrices
            # Handle device mismatch when using device_map='auto'
            if self.use_device_map_auto:
                input_ids_A = input_ids_A.to(self.emb_A.weight.device)
                input_ids_B = input_ids_B.to(self.emb_B.weight.device)
                attention_mask_A = attention_mask_A.to(self.emb_A.weight.device)
                attention_mask_B = attention_mask_B.to(self.emb_B.weight.device)
            embeds_A = self.emb_A(input_ids_A)
            embeds_B = self.emb_B(input_ids_B)
        
        # Apply matrix projections to break symmetry via rotation/transformation
        embeds_A = self.stream_proj_A(embeds_A)
        embeds_B = self.stream_proj_B(embeds_B)
        
        # Fuse embeddings based on objective
        if self.objective == 'mixup' and lambdas is not None:
            # Mixup: weighted combination based on lambda
            # lambdas shape: [batch_size], need to broadcast to [batch_size, seq_len, hidden_dim]
            lambdas_expanded = lambdas.view(-1, 1, 1).to(embeds_A.dtype).to(embeds_A.device)
            fused_embeds = lambdas_expanded * embeds_A + (1 - lambdas_expanded) * embeds_B
            # Note: classic mixup also reduces variance, but we keep it standard for now.
        else:
            # Dynamic Rescaling Strategy: ensure fused embedding has the same norm as inputs
            # This automatically handles different cases (orthogonal vs collinear)
            
            # Compute sum of embeddings
            sum_embeds = embeds_A + embeds_B
            
            # Compute norms along the hidden dimension
            norm_A = torch.norm(embeds_A, p=2, dim=-1, keepdim=True)
            norm_B = torch.norm(embeds_B, p=2, dim=-1, keepdim=True)
            norm_sum = torch.norm(sum_embeds, p=2, dim=-1, keepdim=True)
            
            # Target norm is the average of input norms
            target_norm = (norm_A + norm_B) / 2.0
            
            # Compute scale factor with epsilon for numerical stability
            scale_factor = target_norm / (norm_sum + 1e-6)
            
            # Apply dynamic rescaling
            fused_embeds = sum_embeds * scale_factor
        
        # Use OR operation for attention mask: attend if EITHER stream has valid token
        # This prevents empty attention masks when sequences have different lengths
        attention_mask = torch.maximum(attention_mask_A, attention_mask_B)
        
        transformer_outputs = self.transformer(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = transformer_outputs.last_hidden_state
        
        if self.objective == 'separate' or self.objective == 'separate_distillation':
            # Handle device mismatch when using device_map='auto'
            # Move last_hidden_state to the device of the unembedding layer
            if self.use_device_map_auto:
                last_hidden_state = last_hidden_state.to(self.unemb_A.weight.device)
            logits_A = self.unemb_A(last_hidden_state)
            logits_B = self.unemb_B(last_hidden_state)
            return logits_A, logits_B
        elif self.objective == 'averaged' or self.objective == 'mixup' or self.objective == 'distillation':
            # Handle device mismatch when using device_map='auto'
            if self.use_device_map_auto:
                last_hidden_state = last_hidden_state.to(self.unemb.weight.device)
            logits = self.unemb(last_hidden_state)
            return logits
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    @torch.no_grad()
    def generate(
        self,
        input_ids_A,
        attention_mask_A,
        input_ids_B,
        attention_mask_B,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        Autoregressive generation for dual-stream model.
        
        Args:
            input_ids_A: Input token IDs for stream A [batch_size, seq_len]
            attention_mask_A: Attention mask for stream A [batch_size, seq_len]
            input_ids_B: Input token IDs for stream B [batch_size, seq_len]
            attention_mask_B: Attention mask for stream B [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability (None = disabled)
            top_p: Keep top tokens with cumulative probability >= top_p (None = disabled)
            do_sample: Whether to sample (True) or use greedy decoding (False)
            eos_token_id: End of sequence token ID (stops generation)
            pad_token_id: Padding token ID
            
        Returns:
            If objective == 'separate' or objective == 'separate_distillation':
                generated_A: Generated tokens for stream A [batch_size, seq_len + max_new_tokens]
                generated_B: Generated tokens for stream B [batch_size, seq_len + max_new_tokens]
            Otherwise:
                generated: Generated tokens [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        # Clone inputs to avoid modifying original tensors
        current_ids_A = input_ids_A.clone()
        current_ids_B = input_ids_B.clone()
        current_mask_A = attention_mask_A.clone()
        current_mask_B = attention_mask_B.clone()
        
        batch_size = input_ids_A.size(0)
        device = input_ids_A.device
        
        # Track which sequences have finished generating
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids_A=current_ids_A,
                attention_mask_A=current_mask_A,
                input_ids_B=current_ids_B,
                attention_mask_B=current_mask_B
            )
            
            if self.objective == 'separate' or self.objective == 'separate_distillation':
                logits_A, logits_B = outputs
                # Get logits for last position
                next_token_logits_A = logits_A[:, -1, :] / temperature
                next_token_logits_B = logits_B[:, -1, :] / temperature
            else:
                logits = outputs
                next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    next_token_logits_A = self._top_k_filtering(next_token_logits_A, top_k)
                    next_token_logits_B = self._top_k_filtering(next_token_logits_B, top_k)
                else:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering if specified
            if top_p is not None and top_p < 1.0:
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    next_token_logits_A = self._top_p_filtering(next_token_logits_A, top_p)
                    next_token_logits_B = self._top_p_filtering(next_token_logits_B, top_p)
                else:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample or greedy decode
            if do_sample:
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    probs_A = torch.softmax(next_token_logits_A, dim=-1)
                    probs_B = torch.softmax(next_token_logits_B, dim=-1)
                    next_token_A = torch.multinomial(probs_A, num_samples=1)
                    next_token_B = torch.multinomial(probs_B, num_samples=1)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    next_token_A = torch.argmax(next_token_logits_A, dim=-1, keepdim=True)
                    next_token_B = torch.argmax(next_token_logits_B, dim=-1, keepdim=True)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS tokens
            if eos_token_id is not None:
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    finished = finished | (next_token_A.squeeze(-1) == eos_token_id) | (next_token_B.squeeze(-1) == eos_token_id)
                else:
                    finished = finished | (next_token.squeeze(-1) == eos_token_id)
            
            # Replace tokens for finished sequences with pad token
            if pad_token_id is not None and finished.any():
                if self.objective == 'separate' or self.objective == 'separate_distillation':
                    next_token_A[finished] = pad_token_id
                    next_token_B[finished] = pad_token_id
                else:
                    next_token[finished] = pad_token_id
            
            # Append new tokens
            if self.objective == 'separate' or self.objective == 'separate_distillation':
                current_ids_A = torch.cat([current_ids_A, next_token_A], dim=1)
                current_ids_B = torch.cat([current_ids_B, next_token_B], dim=1)
            else:
                current_ids_A = torch.cat([current_ids_A, next_token], dim=1)
                current_ids_B = torch.cat([current_ids_B, next_token], dim=1)
            
            # Update attention masks
            current_mask_A = torch.cat([current_mask_A, torch.ones((batch_size, 1), dtype=current_mask_A.dtype, device=device)], dim=1)
            current_mask_B = torch.cat([current_mask_B, torch.ones((batch_size, 1), dtype=current_mask_B.dtype, device=device)], dim=1)
            
            # Stop if all sequences are finished
            if eos_token_id is not None and finished.all():
                break
        
        if self.objective == 'separate' or self.objective == 'separate_distillation':
            return current_ids_A, current_ids_B
        else:
            return current_ids_A  # For non-separate objectives, both streams share the same output
    
    def _top_k_filtering(self, logits, top_k):
        """Filter a distribution of logits using top-k filtering."""
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        """Filter a distribution of logits using nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
