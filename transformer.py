import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import wandb


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# remember to move to the appropriate device!


# define class here
# first let's define an MLP

# next step is gonna be creating an embedding.


# TODO: today - let's implement positional encoding (remember the sqrt(d) factor)
# TODO: today - remember you're also doing a system question.



# TODO: finish transformers, with positional encoding. no training yet.
# TODO: review your work, with special emphasis on what you missed.
# TODO: do the hard-written positional encoding.
# then do system design problem.

# then get alexnet working
# then get imdb thingy working (tokenization! use this as chance to use huggingface tokenizer).
# then get CLIP working.

# then do either performance
# or LLM upgrades.




# If you get tired, move to system design problem
# TODO: start training on alexnet and imbd thingy
# TODO: then start training on CLIP and fix it up.
# TODO: hey can you do training loop time measurement, all optimizations, and even batch acculumation
# TODO: do this on like Alexnet or something.
# then come back here and implement the other stuff (ROPE, other masking, and k-v caching)

# prove gradient accumulation to yourself, good thing you're looking into this!



# if you're doing regression - THINK THROUGH whether you need a causal mask and all the width you havein
# a normal transformer.
# think of it for a classification head.

# last things for transformer:
# tying weights DONE
# multiply by embedding dimension. DONE


# TODO: 
# You also need to initialize weights!
# switch to GELU!





class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout_rate, context_len, pad_token_id):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.pad_token_id = pad_token_id

        # cache a boolean causal mask up to context_len
        causal = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal)

    def forward(self, x, input_ids, cos_table, sin_table):
    	#input_ids (B, T)
        B, T, _ = x.shape
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        assert self.head_dim % 2 == 0

	    # ensure broadcast shapes; for (B,T,Dh) use (T,1) broadcasting on last axis pairs
	    # make cos/sin rank-3 so they broadcast across batch (and heads if present)
		cos = cos_table[:T, :self.head_dim//2].unsqueeze(0)  # (1,T,Dh/2)
	    sin = sin_table[:T, :self.head_dim//2].unsqueeze(0)  # (1,T,Dh/2)

	    # split into even and odd dims. They are split into B,T,head_dim//2
	    q1, q2 = q[..., ::2], q[..., 1::2]        # (B,T,Dh/2) each
	    k1, k2 = k[..., ::2], k[..., 1::2]

	    # this intially creates a list of tensors giving the even and odd dimensions.
        # it then stacks them (without combining them, there is an extra dim)
        # then it calls flatten(-2) to interleave them to the desired structure.
	    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
	    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)


		scores = (q_rot @ k_rot.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B,T,T)

        # slice to current T
        causal_mask = self.causal_mask[:T, :T].unsqueeze(0).to(scores.device)  # (1,T, T)
        key_valid = (input_ids != self.pad_token_id).unsqueeze(1) # (B,1,T)
        # we intentionally don't pad queries, they'll be masked later in the loss (otherwise we'll have softmaxes with denominator of 0).
        #query_valid = (input_ids != self.pad_token_id).unsqueeze(-1)  # [B, T, 1]
        keep = causal_mask & key_valid
        scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v
		


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, context_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attention_heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim, dropout_rate, context_len)
             for _ in range(num_heads)]
        )
        self.mha_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha_dropout = nn.Dropout(dropout_rate)

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Linear(embed_dim, ff_dim)
        self.mlp2 = nn.Linear(ff_dim, embed_dim)
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, cos_table, sin_table):                  # x: (B, T, E)
        # MHA (pre-norm)
        ln_x = self.layer_norm_1(x)
        concat = torch.cat([h(ln_x, cos_table, sin_table) for h in self.attention_heads], dim=-1)  # (B, T, E)
        x = x + self.mha_dropout(self.mha_linear(concat))

        # FFN (pre-norm)
        ln_x = self.layer_norm_2(x)
        x = x + self.ff_dropout(self.mlp2(F.gelu(self.mlp1(ln_x))))
        return x


class Transformer(nn.Module):
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


	def __init__(self, vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks):
		super().__init__()
		# two embedding layers? 
		self.embed_dim = embed_dim
		self.embedding = nn.Embedding(vocab_size, embed_dim)


        # cache a boolean causal mask up to context_len

        self.prepare_rope_posemb(context_len, embed_dim//num_heads)


		self.learned_pos_enc = nn.Embedding(context_len,embed_dim)

		self.emb_dropout = nn.Dropout(dropout_rate)
		ff_dim = embed_dim*4
		self.transformer_blocks = nn.Sequential(*[
			TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, context_len) for _ in range(n_blocks)
			])
		self.output_layernorm = nn.LayerNorm(embed_dim)
		self.output = nn.Linear(embed_dim, vocab_size, bias=False)

		# tie weights of embedding layer and output layer
		self.output.weight = self.embedding.weight

	def forward(self, x):
		# x has length B, T
		T = x.shape[1]

        if T > self.context_len:
            raise ValueError(f"seq len {T} exceeds context_len {self.context_len}")

		x = self.embedding(x)*(self.embed_dim ** -0.5) # B,T,E

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
        	x = blk(x, cos, sin)

		x = self.transformer_blocks(x) # B,T,E
		return self.output(self.output_layernorm(x)) # outputting logits - B,T,vocab_size


# Now let's do positional encoding. DONE.
# Then let's do the masking that would be needed for SFT. DONE
# Then, let's do ROPE embeddings. DONE

# Lastly, do k-v caching. Then I thhink you're done but might wanna go through todo list.


# TODO: what should you do about gradient accumulation here?
def transformer_loss(logits,labels,pad_token_id):
	loss_mask = (labels != pad_token_id).float() # (B,T)
	# collapse logits from (B,T,vocab_size) to (B*T,vocab_size) and collapse labels from (B,T) to (B*T)
	# this is done to be able to use the cross_entropy function
	# then we return the shape to B,T
	unreduced_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), labels.view(-1), reduction="none").view(B,T)
	return (unreduced_loss * loss_mask).sum()/loss_mask.sum() # average over only non-padded losses.


def main():
	config = {
		"weight_decay": 0.1,
		"learning_rate": 0.001,
		"neuron_size": 200
	}


	batch_size = 16
	num_epochs = 3


	transform = transforms.Compose(
	    [transforms.ToTensor(),
	    transforms.Normalize((0.5,), (0.5,))])

	# Create datasets for training & validation, download if necessary
	training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
	validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

	# Create data loaders for our datasets; shuffle for training, not for validation
	training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

	# Report split sizes
	print('Training set has {} instances'.format(len(training_set)))
	print('Validation set has {} instances'.format(len(validation_set)))

	# can we download the finewebedu set, a little sample?
	# define dataloader here
	print(f'Training set has shape {training_set[0][0].shape}')

	with wandb.init(config=config,project="mnist-playground",entity="alancasallas-self") as run:

		model = MLP(wandb.config.neuron_size)
		model.to(device)

		# todo: can you implement batch accumulation?

		# let's try Adam for now with default parameters
		optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
		

		for epoch in range(num_epochs):
			metrics = {"epoch": epoch}
			print(f"EPOCH {epoch}")
			train_losses = 0
			train_total = 0

			model.train()
			for i,data in enumerate(training_loader):
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				outputs = model(inputs) # outputs are logits (B,T,vocab_size)
				loss = transformer_loss(outputs,labels) # labels are B,T

				train_losses += loss.item()*inputs.size(0)
				train_total += inputs.size(0)
				loss.backward()
				optimizer.step()
			print(f"train loss {train_losses/train_total}")
			metrics.update({"train_loss": train_losses/train_total})


			# remember model.train() and model.eval()!
			model.eval()
			val_losses = 0
			val_correct = 0
			val_total = 0

			with torch.no_grad():
				for i,data in enumerate(validation_loader):
					inputs, labels = data
					inputs = inputs.to(device)
					labels = labels.to(device)
					outputs = model(inputs) # this doesn't work because outputs is a 10 vector!
					_, predicted = torch.max(outputs, 1)
					loss =loss_fn(outputs,labels)
					val_losses += loss.item()*inputs.size(0)
					val_correct += (labels==predicted).sum().item()
					val_total += labels.size(0)

			print(f"validation loss {val_losses/val_total} accuracy {val_correct/val_total}")
			metrics.update({"val_loss": train_losses/train_total})
			wandb.log(metrics)




if __name__ == "__main__":
	main()

