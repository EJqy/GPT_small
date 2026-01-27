# pip install regex
import regex as re

def split_with_special_tokens(text, special_tokens):
    if special_tokens:
        # join together as "|" splitted string
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        
        # split into segments (keep tokens due to capture group)
        parts = re.split(f"({special_regex})", text)
        
        train_segments = [p for p in parts if (p not in special_tokens)]

        return train_segments
    return [text]

if __name__ == "__main__":
    text = "Hello <BOS> world! [SEP] 你好🙂 <EOS> done."
    special_tokens = ["<BOS>", '[SEP]', '<EOS>']

    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    parts = split_with_special_tokens(text, special_tokens)
    for segment in parts:
        words = gpt2_pat.findall(segment)
        print(words)



