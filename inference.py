import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

from transformer import Transformer
import common_utils


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

MAX_RESPONSE_LEN = 300 # in tokens
TOP_P = None
TOP_K = None
TEMPERATURE = 0.6

# for now, you will just take ONE SAMPLE, and stream tokens back (figure out how that works with Gradio.)
def run_inference(model, system_string, conversation_strings):
	tok = PreTrainedTokenizerFast.from_pretrained("./casallm_bpe")
	print(f"loaded tokenizer with a {tok.vocab_size} vocab size.")
	STOP_TOKEN = tok.eos_token_id

	# TODO: you have to import this function from somewhere else
	tokens, _ = encode_chat_to_ids_and_mask([{"role":"system","content": system_string}]+conversation_strings,tok)

	# TODO: put temperature in here. What about top-p and top-k?

	# TODO: streaming back tokens is gonna be important, do that soon.
	# TODO: first, consult on how to do it (queues back to fastapi process? then do it.)



	stop_token_found = False
	num_tokens_generated = 0
	while not tokens[-1] != STOP_TOKEN and num_tokens_generated < MAX_RESPONSE_LEN:

		# run inference forever
		new_batch = input("Give me a new batch of data")

		# TODO: vital! the tokens above include the assistant response, we DO NOT want that.

		token_batch = tokens.unsqueeze(0) # (T) -> (B, T)

		logits = model(tokens) # (B,T,V)
		num_tokens_generated += 1

		logits = logits.squeeze(0)[-1,:] # # (B,T,V) -> (V) (take last element in sequence)
		probs = F.softmax(logits/TEMPERATURE) # (V)
		
		# TODO: implement top-k and top-p

		token = torch.multinomial(probs,samples=1)

		tokens = torch.cat((tokens,token),dim=0)



if __name__ == "__main__":
	model_checkpoint_path = None # TODO: put in a path
	ckpt = torch.load(path, map_location=str(device))
    model.load_state_dict(ckpt["model"])

	# load model and run inference
	system_str = "You are an AI assistant. Please give concise and helpful answers."

	# this is gonna be sent to the REST API.
	# TODO: design system. Will we pass in entire conversation every time, or cache/save conversations server-side?
	conversation_strs = [
		{
		"role": "user",
		"content": "who are you?"
		}
	]
	run_inference(system_str, conversation_strs)

