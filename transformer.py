import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb


# create seven files:
# inference
# pre-training
# sft
# dpo
# bpe_trainer
# pre-processing (tokenize the datasets)
# but also: get the datasets for SFT training and DPO training. Yo ucan use the kangaroo ones but also take a quick look at others.
# you're gonna choose formats for all of them.



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        self.QKV_proj = nn.Linear(embed_dim, embed_dim*3, bias=False)

        self.attn_head_dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id

        # cache a boolean causal mask up to context_len
        causal = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

        self.mha_dropout = nn.Dropout(dropout_rate)
        self.mh_linear = nn.Linear(embed_dim, embed_dim)
        self.mh_linear.is_res_init_scaling_needed = True # this is a flag for initialization to do special scaling.

    def apply_rope_(self, x, cos, sin):
        # split into even and odd dims. They are split into B,nh,T,head_dim//2
        x1 = x[..., ::2] # (B,nh,T,Dh/2) each
        x2 = x[..., 1::2]
	    # creates a list of tensors giving the even and odd dimensions, then calls flatten(-2) to interleave them to the desired structure.
        # resulting dim: (B,nh,T,Dh)
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return x_rot


    def forward(self, x, input_ids, cos_table, sin_table):
    	#input_ids (B, T)
        B, T, embed_dim = x.shape

        qkv = self.QKV_proj(x) # shape B,T,embed_dim*3 (this is qkv combined for all heads)
        q,k,v = qkv.split(self.embed_dim,dim=-1)
        # create a number_of_heads dim and move it to the front (because attention needs last two dim to be T and head_dim)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        # ensure causal mask covers T
        if T > self.causal_mask.size(0):
            raise ValueError(f"sequence length {T} exceeds cached context_len {self.causal_mask.size(0)}")

        assert self.head_dim % 2 == 0

        # unsqueeze twice into: (1, 1, T, Dh/2)
        cos = cos_table[:T, :Dh//2].to(x.device, x.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin_table[:T, :Dh//2].to(x.device, x.dtype).unsqueeze(0).unsqueeze(0)
	    
        # apply RoPE to Q and K
        q = self.apply_rope_(q, cos, sin) # (B, H, T, Dh)
        k = self.apply_rope_(k, cos, sin) # (B, H, T, Dh)

        causal_mask = self.causal_mask[:T, :T].to(x.device).unsqueeze(0).unsqueeze(0)  # causal: (1, 1, T, T)
        # we only mask keys. queries be masked later, in the loss (otherwise we'll have softmaxes with denominator of 0).
        key_keep = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2) # key padding: (B, 1, 1, T)
        attn_mask = causal_mask & key_valid # (B, 1, T, T)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_head_dropout_rate if self.training else 0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, embed_dim) # (B,nh,T,d_h) -> (B,T,embed_dim) effectivley concatenating the heads

        return self.mha_dropout(self.mha_linear(y))
		


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout_rate, context_len, pad_token_id)

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Linear(embed_dim, embed_dim*4)
        self.mlp2 = nn.Linear(ff_dim, embed_dim)
        self.mlp2.is_res_init_scaling_needed = True
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, input_ids, cos_table, sin_table):
        # MHA 
        ln_x = self.layer_norm_1(x)
        x = x + self.mha(ln_x, input_ids, cos_table, sin_table)

        # FFN
        ln_x = self.layer_norm_2(x)
        x = x + self.ff_dropout(self.mlp2(F.gelu(self.mlp1(ln_x))))
        return x


class Transformer(nn.Module):
	def __init__(self, vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks, pad_token_id):
		super().__init__()
		self.embed_dim = embed_dim
		self.embedding = nn.Embedding(vocab_size, embed_dim)

        # cache a boolean causal mask up to context_len
        self.prepare_rope_posemb(context_len, embed_dim//num_heads)

		#self.learned_pos_enc = nn.Embedding(context_len,embed_dim)

		self.emb_dropout = nn.Dropout(dropout_rate)
		
		self.transformer_blocks = nn.Sequential(*[
			TransformerBlock(embed_dim, num_heads, dropout_rate, context_len, pad_token_id) for _ in range(n_blocks)
			])
		self.output_layernorm = nn.LayerNorm(embed_dim)
		self.output = nn.Linear(embed_dim, vocab_size, bias=False)

		# tie weights of embedding layer and output layer
		self.output.weight = self.embedding.weight
		self.apply(self.init_weights)

	def prepare_classical_posemb(self, context_len, embed_dim):
        # prepare classical positional encoding
        # TODO: move to its own function?
		pos = torch.arange(context_len, dtype=torch.float32) # (T,)
		inv_freq = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)) # (E/2,)
		# outer product: (T, 1) * (1, E/2) -> (T, E/2)
		angles = pos[:, None] * inv_freq[None, :]                                  # (T, E/2)
		pe = torch.zeros(context_len, embed_dim, dtype=torch.float32)              # (T, E)
		pe[:, 0::2] = torch.sin(angles)                                           # even dims
		pe[:, 1::2] = torch.cos(angles)                                           # odd dims
		# store as (1, T, E) for broadcast to batch
		self.register_buffer("classical_positional_encoding", pe.unsqueeze(0), persistent=False)
		

	def init_weights(self, module):
		# linear layers get init to std 0.02, but biases get init to 0.0
		# this function is inspired by the nanogpt implementation
		if isinstance(module, nn.Linear):
			std = 0.02
			if hasattr(module, is_res_init_scaling_needed):
				std = std * ((2 * self.n_blocks) ** -0.5)
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def prepare_rope_posemb(self, context_len, attn_head_dim):
		pos = torch.arange(context_len, dtype=torch.float32) # (T,)
		inv_freq = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / attn_head_dim)) # (E/2,)
		# outer product: (T, 1) * (1, E/2) -> (T, E/2)
		angles = pos[:, None] * inv_freq[None, :]                                  # (T, E/2)
		cosines = torch.sin(angles)                                           # even dims
		sines = torch.cos(angles)                                           # odd dims
		# store as (1, T, E) for broadcast to batch
		self.register_buffer("rope_cosines", pe.unsqueeze(0), persistent=False)
		self.register_buffer("rope_sines", pe.unsqueeze(0), persistent=False)

	def forward(self, input_ids):
		# x has length B, T
		T = x.shape[1]

        if T > self.context_len:
            raise ValueError(f"seq len {T} exceeds context_len {self.context_len}")

		x = self.embedding(input_ids)*(self.embed_dim ** -0.5) # B,T,E

		# TODO: consider putting the to inside the attention head to use same dtype as q,k.
		cos = self.rope_cosines[:T].to(torch.get_default_dtype())  # (T, Dh/2)
		sin = self.rope_sines[:T].to(torch.get_default_dtype())  # (T, Dh/2)

		# below would be code to use classical positional encoding
		# TODO: we'll likely get rid of this.
		### pos_emb = self.classical_positional_encoding[:, :T, :] # (1, T, E)
		### x = tok_emb + pos_emb.to(tok_emb.dtype)  # to(dtype) was used in case we are using FP16 or other types during training.                           

        #pos = torch.arange(T, device=x.device)               # (T,)
        #pos = self.learned_pos_enc(pos).unsqueeze(0)             # (1, T, E)
        #x = self.emb_dropout(tok + pos) 

        x = self.emb_dropout(x)
        for blk in self.transformer_blocks:
        	x = blk(x, input_ids, cos, sin)

		x = self.transformer_blocks(x) # B,T,E
		return self.output(self.output_layernorm(x)) # outputting logits - B,T,vocab_size




