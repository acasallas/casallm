import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# TODO: explore the finewebedu dataset
# TODO: can you see if you need to add a token when a dataset ends
# TODO: and of course the whole reason we are doing this - special tokens for prompts!


SEED = 42
VOCAB_SIZE = 25_000

def main():
    ds = load_dataset("pixparse/cc3m-wds", split="train")
    ds = ds.shuffle(seed=SEED)  # avoid head-of-corpus bias
    n = ds.num_rows

    # Be tolerant to different column names
    candidates = ["txt", "caption", "text", "description"]
    text_col = next((c for c in candidates if c in ds.column_names), None)
    if text_col is None:
        raise ValueError(f"Couldn't find a text column in: {ds.column_names}")

    def batch_iterator(batch_size=1000):
        batch = []
        for ex in ds:
            t = ex[text_col]
            if t:
                batch.append(t)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Model + pretokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["[PAD]", "[EOS]", "[UNK]"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    print("Starting BPE training…")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=n)

    # Now IDs exist → add post-processing
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[("[EOS]", eos_id)],
    )

    os.makedirs("clip_bpe", exist_ok=True)
    tokenizer.save("clip_bpe/tokenizer.json")

    raise ValueError("Remember to save the special hugging face tokenizer as well!")


if __name__ == "__main__":
    main()