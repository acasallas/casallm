import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# export TOKENIZERS_PARALLELISM=true

SEED = 42
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>"]
VOCAB_SIZE = 40_960  # includes special tokens
JOIN_BATCH = 1_000

def main():
    remote_name = "sample-10BT"
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # drop extra columns to help Arrow -> Python work.
    text_col = "text"
    drop_cols = [c for c in ds.column_names if c != text_col]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    n = ds.num_rows

    def big_chunk_iterator(batch_size=JOIN_BATCH):
        batch = []
        append = batch.append
        join = "\n".join
        for ex in ds:
            t = ex.get(text_col)
            if t:
                append(t)
            if len(batch) >= batch_size:
                yield join(batch)
                batch.clear()
        if batch:
            yield join(batch)

    tokenizer = Tokenizer(BPE(unk_token=None))  # byte-level BPE => no <unk>
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,           # total size INCLUDING specials
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Starting BPE training…")
    # no `length=` -> avoids progress bookkeeping cost & mismatches
    tokenizer.train_from_iterator(big_chunk_iterator(), trainer=trainer, length=n // JOIN_BATCH)

    # BOS/EOS via post-processor
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    os.makedirs("casallm_bpe", exist_ok=True)
    tokenizer.save("casallm_bpe/tokenizer.json")

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
        model_max_length=1024,
        padding_side="right",
        truncation_side="right",
    )
    hf_tok.save_pretrained("casallm_bpe")
    print("Saved to ./casallm_bpe")

if __name__ == "__main__":
    main()
