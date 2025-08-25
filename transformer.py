import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb


# note for inference there's two steps: a prefill (that generates a kv cache) and a decode step (where you use the kv cachce)

# Notes:
# a kv cache should be passed into forward, along with a use_cache flag
# a request is typically prefill (no_cache, but a cache is created) or decode (where we pass it one by one using the cache)
# we'll pass in batches with their own row in the kv cache, the forward pass will pad to the longest request and 

"""
recommended pattern for kv caching:
def forward(self, x, *, past_kv=None, use_cache=False, **kw):
    if use_cache:
        with torch.inference_mode():   # crucial: no autograd through cache
            return self._forward_decode_cached(x, past_kv, **kw)
    else:
        return self._forward_prefill(x, **kw)  # normal training path
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.QKV_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.mh_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mh_linear.is_res_init_scaling_needed = True  # flag for scaled init
        self.mha_dropout = nn.Dropout(dropout_rate)

        self.pad_token_id = pad_token_id
        self.attn_head_dropout_rate = dropout_rate

        # cached lower-triangular causal mask (True on allowed lower triangle)
        causal = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: (B, H, T, Dh), cos/sin: (1, 1, T, Dh/2)
        # split interleaved even/odd last-dim
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        # rotate
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        # interleave back
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(self, x, input_ids, cos_table, sin_table):
        # x: (B, T, E), input_ids: (B, T)
        B, T, E = x.shape
        if T > self.causal_mask.size(0):
            raise ValueError(f"sequence length {T} exceeds cached context_len {self.causal_mask.size(0)}")
        assert self.head_dim % 2 == 0

        qkv = self.QKV_proj(x)  # (B, T, 3E)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # (B, H, T, Dh)
        # create a number_of_heads dim and move it to the front (because attention needs last two dim to be T and head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE: slice and cast to match q/k dtype & device
        cos = cos_table[:T, : (self.head_dim // 2)].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin_table[:T, : (self.head_dim // 2)].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Build attention bias: (B, 1, T, T) -> broadcasts over heads
        # Start with causal lower triangle; mask (set -inf) outside it.
        attn_bias = torch.zeros((B, 1, T, T), dtype=x.dtype, device=x.device)
        causal = self.causal_mask[:T, :T]  # (T, T) bool
        attn_bias = attn_bias.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # we only mask keys. queries be masked later, in the loss (otherwise we'll have softmaxes with denominator of 0).
        key_keep = (input_ids != self.pad_token_id).to(x.device)  # (B, T) bool
        attn_bias = attn_bias.masked_fill(~key_keep[:, None, None, :], float("-inf"))

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,                  # additive mask
            dropout_p=self.attn_head_dropout_rate if self.training else 0.0,
            is_causal=False
        )

        # concat heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.mh_linear(y)
        return self.mha_dropout(y)

        
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout_rate, context_len, pad_token_id)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Linear(embed_dim, embed_dim * 4, bias=True)
        self.mlp2 = nn.Linear(embed_dim * 4, embed_dim, bias=True)
        self.mlp2.is_res_init_scaling_needed = True
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, input_ids, cos_table, sin_table):
        # multihead attention layer
        x = x + self.mha(self.ln1(x), input_ids, cos_table, sin_table)
        # feed forward section
        x = x + self.ff_dropout(self.mlp2(F.gelu(self.mlp1(self.ln2(x)))))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks, pad_token_id):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.context_len = context_len
        self.head_dim = embed_dim // num_heads

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # RoPE tables: (T, Dh/2)
        self._prepare_rope_tables(context_len, self.head_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout_rate, context_len, pad_token_id)
            for _ in range(n_blocks)
        ])
        self.output_layernorm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights
        self.output.weight = self.embedding.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # linear layers get init to std 0.02, but biases get init to 0.0
        # this function is inspired by the nanogpt implementation
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "is_res_init_scaling_needed"):
                std = std * ((2 * self.n_blocks) ** -0.5)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_rope_tables(self, context_len, head_dim):
        # Standard RoPE theta = 10000.0
        half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))   # (Dh/2,)
        t = torch.arange(context_len, dtype=torch.float32)                                 # (T,)
        freqs = torch.outer(t, inv_freq)                                                   # (T, Dh/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("rope_cos_table", cos, persistent=False)  # (T, Dh/2)
        self.register_buffer("rope_sin_table", sin, persistent=False)  # (T, Dh/2)

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        if T > self.context_len:
            raise ValueError(f"seq len {T} exceeds context_len {self.context_len}")

        x = self.embedding(input_ids) * math.sqrt(self.embed_dim)  # (B, T, E)

        # Match q/k dtype automatically by using x.dtype (autocast-friendly)
        cos = self.rope_cos_table[:T].to(dtype=x.dtype)
        sin = self.rope_sin_table[:T].to(dtype=x.dtype)

        x = self.emb_dropout(x)
        for blk in self.transformer_blocks:
            x = blk(x, input_ids, cos, sin)

        x = self.output_layernorm(x)
        logits = self.output(x)  # (B, T, vocab_size)
        return logits
