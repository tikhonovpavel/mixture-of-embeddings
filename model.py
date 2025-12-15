import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig


class DualStreamLLM(nn.Module):
    def __init__(self, model_name, freeze_backbone=True,
                 embedding_init_strategy='random', embedding_init_noise_std=0.01,
                 unembedding_init_strategy=None,
                 objective='separate', shared_embeddings=False):
        super().__init__()
        self.objective = objective
        self.shared_embeddings = shared_embeddings
        
        if unembedding_init_strategy is None:
            unembedding_init_strategy = embedding_init_strategy
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            use_cache=False
        )
        
        base_model_config = AutoConfig.from_pretrained(model_name)
        vocab_size = base_model_config.vocab_size
        hidden_dim = base_model_config.hidden_size
        
        # Embedding initialization
        if shared_embeddings:
            # Single shared embedding matrix for both streams
            self.emb = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        else:
            # Separate embedding matrices
            self.emb_A = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
            self.emb_B = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        
        # Unembedding initialization
        if objective == 'separate':
            self.unemb_A = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
            self.unemb_B = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        elif objective == 'averaged' or objective == 'mixup' or objective == 'distillation':
            self.unemb = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
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
        if objective == 'separate':
            _initialize_weights(self.unemb_A, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)
            # For B, we add noise only if the strategy is 'noisy_copy'.
            if unembedding_init_strategy == 'noisy_copy':
                _initialize_weights(self.unemb_B, original_lm_head_weights, 'noisy_copy', embedding_init_noise_std)
            else:
                _initialize_weights(self.unemb_B, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)
        else: # 'averaged' or 'mixup' objective
            _initialize_weights(self.unemb, original_lm_head_weights, unembedding_init_strategy, embedding_init_noise_std)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if hasattr(self.backbone, 'model'):
            self.transformer = self.backbone.model
        elif hasattr(self.backbone, 'transformer'):
            self.transformer = self.backbone.transformer
        elif hasattr(self.backbone, 'gpt_neox'):
            self.transformer = self.backbone.gpt_neox
        else:
            raise AttributeError("The backbone model does not have a standard transformer attribute ('model', 'transformer', 'gpt_neox').")
        
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, lambdas=None):
        # Get embeddings
        if self.shared_embeddings:
            # Use same embedding matrix for both streams
            embeds_A = self.emb(input_ids_A)
            embeds_B = self.emb(input_ids_B)
        else:
            # Use separate embedding matrices
            embeds_A = self.emb_A(input_ids_A)
            embeds_B = self.emb_B(input_ids_B)
        
        # Fuse embeddings based on objective
        if self.objective == 'mixup' and lambdas is not None:
            # Mixup: weighted combination based on lambda
            # lambdas shape: [batch_size], need to broadcast to [batch_size, seq_len, hidden_dim]
            lambdas_expanded = lambdas.view(-1, 1, 1).to(embeds_A.dtype).to(embeds_A.device)
            fused_embeds = lambdas_expanded * embeds_A + (1 - lambdas_expanded) * embeds_B
        else:
            # Average fusion for other objectives
            fused_embeds = (embeds_A + embeds_B) * 0.5
        
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
        
        if self.objective == 'separate':
            logits_A = self.unemb_A(last_hidden_state)
            logits_B = self.unemb_B(last_hidden_state)
            return logits_A, logits_B
        elif self.objective == 'averaged' or self.objective == 'mixup' or self.objective == 'distillation':
            logits = self.unemb(last_hidden_state)
            return logits
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
