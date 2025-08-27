import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
kv cache allocation:

# Example (do this once per layer before decode):
B = batch_size
H = num_heads
Dh = embed_dim // num_heads
Tmax = context_len
dtype = model.embedding.weight.dtype
device = model.embedding.weight.device

k_cache = [torch.empty(B, H, Tmax, Dh, dtype=dtype, device=device) for _ in range(n_blocks)]
v_cache = [torch.empty(B, H, Tmax, Dh, dtype=dtype, device=device) for _ in range(n_blocks)]
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id, use_attn_mask):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_attn_mask = use_attn_mask

        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even"

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
        # x: (B, H, T, Dh), cos/sin: (B, 1, T, Dh/2) or (B,1,1,Dh/2)
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    @staticmethod
    def _rope_apply_per_pos(q_or_k, pos_idx, cos_table, sin_table):
        """
        q_or_k: (B, H, T, Dh)
        pos_idx: (B, T) for prefill OR (B,) for decode (T=1)
        cos_table/sin_table: (T_max, Dh/2) FULL tables
        """
        B, H, T, Dh = q_or_k.shape
        half = Dh // 2
        if pos_idx.dim() == 1:
            # (B,) -> (B,1)
            pos_idx = pos_idx.unsqueeze(1)
        # gather cos/sin per (b,t)
        flat = pos_idx.reshape(-1)  # (B*T,)
        cos = cos_table.index_select(0, flat).view(B, pos_idx.size(1), half)  # (B,T,half)
        sin = sin_table.index_select(0, flat).view(B, pos_idx.size(1), half)
        # reshape to (B,1,T,half) to broadcast over heads
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return MultiHeadAttention._apply_rope(
            q_or_k,
            cos.to(device=q_or_k.device, dtype=q_or_k.dtype),
            sin.to(device=q_or_k.device, dtype=q_or_k.dtype),
        )

    def forward(self, x, input_ids, cos_table, sin_table, use_cache, pos, k_cache, v_cache, cache_kv: bool):
        # x: (B,T,E). For prefill T=max seq len; for decode T=1
        # input_ids: (B,T)
        # pos: when use_cache=True, it's the current per-sample position (cache length) BEFORE writing the new token; shape (B,)
        # k_cache/v_cache: (B, H, T_max, Dh)
        B, T, E = x.shape
        H, Dh = self.num_heads, self.head_dim
        device = x.device

        qkv = self.QKV_proj(x)                                  # (B, T, 3E)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)                 # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        if not use_cache:
            # ---------- PREFILL (T>=1) ----------
            pos_mat = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
            q = self._rope_apply_per_pos(q, pos_mat, cos_table, sin_table)
            k = self._rope_apply_per_pos(k, pos_mat, cos_table, sin_table)

            # Optionally fill caches (for inference prefill)
            if cache_kv:
                # If sequences are padded, you can still write all T and rely on masking;
                # or compute true lengths L and only write [:L_b]
                k_cache[:, :, :T, :] = k
                v_cache[:, :, :T, :] = v

            if self.use_attn_mask:
                attn_bias = torch.zeros((B, 1, T, T), dtype=x.dtype, device=device)
                causal = self.causal_mask[:T, :T]
                attn_bias = attn_bias.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                key_keep = (input_ids != self.pad_token_id)  # (B,T)
                attn_bias = attn_bias.masked_fill(~key_keep[:, None, None, :], float("-inf"))

                y = F.scaled_dot_product_attention(
                    q, k, v,                       # use local k,v; fast and avoids extra reads
                    attn_mask=attn_bias,
                    dropout_p=self.attn_head_dropout_rate if self.training else 0.0,
                    is_causal=False
                )
            else:
                # fast path for training prefill
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_head_dropout_rate if self.training else 0.0,
                    is_causal=True
                )
        else:
            # ---------- DECODE (T==1) ----------
            assert T == 1, "With use_cache=True, pass only the next token per sample as shape (B,1)."
            pos = pos.to(device=x.device, dtype=torch.long)
            assert pos.numel() == B
            assert pos.max().item() < k_cache.size(2), "pos exceeds cache time dimension"

            q = self._rope_apply_per_pos(q, pos, cos_table, sin_table)
            k = self._rope_apply_per_pos(k, pos, cos_table, sin_table)

            b_idx = torch.arange(B, device=device)
            k_cache[b_idx, :, pos, :] = k.squeeze(2)
            v_cache[b_idx, :, pos, :] = v.squeeze(2)

            Tmax_now = int(pos.max().item()) + 1
            K = k_cache[:, :, :Tmax_now, :]
            V = v_cache[:, :, :Tmax_now, :]

            attn_bias = torch.zeros((B, 1, 1, Tmax_now), dtype=x.dtype, device=device)
            key_pos = torch.arange(Tmax_now, device=device).unsqueeze(0)
            causal_mask = key_pos > pos.unsqueeze(1)
            attn_bias = attn_bias.masked_fill(causal_mask[:, None, None, :], float("-inf"))

            y = F.scaled_dot_product_attention(
                q, K, V,
                attn_mask=attn_bias,
                dropout_p=self.attn_head_dropout_rate if self.training else 0.0,
                is_causal=False
            )

        # concat heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.mh_linear(y)
        return self.mha_dropout(y)

        
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id, use_attn_mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout_rate, context_len, pad_token_id, use_attn_mask)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Linear(embed_dim, embed_dim * 4, bias=True)
        self.mlp2 = nn.Linear(embed_dim * 4, embed_dim, bias=True)
        self.mlp2.is_res_init_scaling_needed = True
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, input_ids, cos_table, sin_table, use_cache, pos, k_cache, v_cache, cache_kv):
        x = x + self.mha(self.ln1(x), input_ids, cos_table, sin_table, use_cache, pos, k_cache, v_cache, cache_kv)
        x = x + self.ff_dropout(self.mlp2(F.gelu(self.ln2(x))))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks, pad_token_id, is_pretraining):
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
            TransformerBlock(embed_dim, num_heads, dropout_rate, context_len, pad_token_id, not is_pretraining) # when pretraining we can use SDPA's fast path rather than build our own mask
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

    def forward(self, input_ids, use_cache, pos, k_cache, v_cache, cache_kv: bool = False):
        B, T = input_ids.shape
        if T > self.context_len:
            raise ValueError(f"seq len {T} exceeds context_len {self.context_len}")

        x = self.embedding(input_ids)
        cos = self.rope_cos_table.to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin_table.to(device=x.device, dtype=x.dtype)

        x = self.emb_dropout(x)
        for n, blk in enumerate(self.transformer_blocks):
            x = blk(x, input_ids, cos, sin, use_cache, pos, k_cache[n], v_cache[n], cache_kv)

        x = self.output_layernorm(x)
        logits = self.output(x)
        return logits

