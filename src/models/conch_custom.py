import torch
from torch import nn
from torch.nn import functional as F
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

# Initialize tokenizer
tokenizer = get_tokenizer()


class TextEncoder(nn.Module):
    """Text encoder module for CONCH model
    
    This class encodes text prompts into embeddings using the CONCH transformer.
    """
    
    def __init__(self, clip_model):
        """Initialize text encoder with CLIP model components
        
        Args:
            clip_model: The CLIP model to use for encoding
        """
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.ln_final.weight.dtype
        self.cls_emb = clip_model.cls_emb

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

    def build_attention_mask(self):
        """Build a causal attention mask for transformer
        
        Returns:
            Attention mask tensor
        """
        # Lazily create causal attention mask, with full attention between the tokens
        # PyTorch uses additive attention mask; fill with -inf
        mask = torch.empty(128, 128)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _repeat(self, t, N: int):
        """Repeat tensor along a new dimension
        
        Args:
            t: Input tensor
            N: Number of repetitions
            
        Returns:
            Repeated tensor
        """
        return t.reshape(1, 1, -1).repeat(N, 1, 1)
    
    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        """Build attention mask for CLS token
        
        Args:
            text: Input text tensor
            cast_dtype: Target dtype for the mask
            
        Returns:
            Attention mask for CLS token
        """
        cls_mask = (text != 0).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, 12, 0)
        return additive_mask

    def forward(self, prompts, tokenized_prompts):
        """Forward pass through the text encoder
        
        Args:
            prompts: Input prompts
            tokenized_prompts: Tokenized prompts
            
        Returns:
            Normalized text embeddings
        """
        # Process prompts
        prompts = prompts[:, :-1]
        seq_len = prompts.shape[1]
        attn_mask = self.attn_mask
        seq_len += 1
        
        # Concatenate with CLS embedding
        prompts = torch.cat([prompts, self._repeat(self.cls_emb, prompts.shape[0])], dim=1)
        
        # Build attention mask
        cls_mask = self.build_cls_mask(tokenized_prompts[:, :-1], self.dtype)
        attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        # Forward through transformer
        x = prompts + self.positional_embedding[:seq_len].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Get pooled representation
        pooled, tokens = x[:, -1], x[:, :-1]
        pooled = self.ln_final(pooled)
        x = pooled @ self.text_projection

        # Normalize output
        x = F.normalize(x, dim=-1)
        return x


class PromptLearner(nn.Module):
    """Learnable prompt module for CONCH
    
    This class implements learnable prompts for the CONCH model
    to enable adapting it to specific downstream tasks.
    """
    
    def __init__(self, classnames, clip_model, n_ctx, n_flp=0, num_patch_prompt=0, is_shared=False):
        """Initialize prompt learner
        
        Args:
            classnames: List of class names
            clip_model: CLIP model
            n_ctx: Number of context tokens
            n_flp: Number of fully learnable prompts
            num_patch_prompt: Number of patch prompts
            is_shared: Whether to share prompts across classes
        """
        super().__init__()
        n_cls = len(classnames)
        
        self.n_flp = n_flp
        self.num_patch_prompt = num_patch_prompt

        # Get model parameters
        dtype = clip_model.ln_final.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Initialize context vectors
        if not is_shared:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            if n_flp > 0 and num_patch_prompt > 0:
                flp_vectors = torch.empty(int(n_cls/num_patch_prompt)*n_flp, 75, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            if n_flp > 0 and num_patch_prompt > 0:
                flp_vectors = torch.empty(75, ctx_dim, dtype=dtype)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [(tokenize(texts=[name], tokenizer=tokenizer) > 0).sum() for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        
        # Tokenize prompts
        tokenized_prompts = torch.cat([tokenize(texts=[p], tokenizer=tokenizer) for p in prompts]).to('cuda')
        
        # Get token embeddings
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # Register buffers for token embeddings
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # Initialize fully learnable prompts if needed
        if n_flp > 0 and num_patch_prompt > 0:
            nn.init.normal_(flp_vectors, std=0.02)
            self.flp = nn.Parameter(flp_vectors)
            flp_prefix = " ".join(["X"] * 75)
            tokenized_flp = torch.cat([tokenize(texts=[flp_prefix+"."], tokenizer=tokenizer) 
                                      for _ in range(int(n_cls/num_patch_prompt)*n_flp)]).to('cuda')
            
            with torch.no_grad():
                embedding_flp = clip_model.token_embedding(tokenized_flp).type(dtype)
            
            self.register_buffer("flp_token_prefix", embedding_flp[:, :1, :])  # SOS
            self.register_buffer("flp_token_suffix", embedding_flp[:, 1 + 75 :, :])  # CLS, EOS
            
            # Combine prompts
            tokenized_prompts_ = []
            for i in range(n_cls):
                if i % num_patch_prompt == 0:
                    cur_i_ = int(i/num_patch_prompt)
                    tokenized_prompts_.append(tokenized_flp[cur_i_:cur_i_+n_flp])
                tokenized_prompts_.append(tokenized_prompts[i].unsqueeze(0))
            self.tokenized_prompts = torch.cat(tokenized_prompts_, dim=0).to('cuda')
        else:
            self.tokenized_prompts = tokenized_prompts
        
        # Store class information
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = "pre"

    def forward(self):
        """Forward pass of the prompt learner
        
        Returns:
            Prompts tensor combining context vectors and token embeddings
        """
        ctx = self.ctx
        
        # Handle fully learnable prompts
        if self.n_flp > 0 and self.num_patch_prompt > 0:
            flp_prefix = self.flp_token_prefix
            flp_suffix = self.flp_token_suffix
            flp = self.flp
            if flp.dim() == 2:
                flp = flp.unsqueeze(0).expand(int(self.n_cls/self.num_patch_prompt)*self.n_flp, -1, -1)
            
        # Expand context if needed
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Get token embeddings
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # Build prompts
        prompts = []
        for i in range(self.n_cls):
            # Add fully learnable prompts if needed
            if self.n_flp > 0 and self.num_patch_prompt > 0:
                if i % self.num_patch_prompt == 0:
                    cur_i_ = int(i/self.num_patch_prompt)
                    flp_i = flp[cur_i_: cur_i_+self.n_flp, :, :]
                    flp_prefix_i = flp_prefix[cur_i_: cur_i_+self.n_flp, :, :]
                    flp_suffix_i = flp_suffix[cur_i_: cur_i_+self.n_flp, :, :]
                    
                    prompt_flp = torch.cat(
                            [
                                flp_prefix_i,
                                flp_i,
                                flp_suffix_i
                            ],
                            dim=1,
                            ) 
                    
                    prompts.append(prompt_flp)
            
            # Add context prompts
            name_len = self.name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            suffix_i = suffix[i : i + 1, name_len:, :]
            ctx_i = ctx[i : i + 1, :, :]
            
            prompt_i = torch.cat(
                [
                    prefix_i,    # (1, 1, dim)
                    ctx_i,       # (1, n_ctx, dim)
                    class_i,     # (1, name_len, dim)
                    suffix_i,    # (1, *, dim) 
                ],
                dim=1,
            )
            prompts.append(prompt_i)
            
        prompts = torch.cat(prompts, dim=0)
        return prompts
