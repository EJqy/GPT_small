import os
from collections import defaultdict, Counter
import regex as re # regular expression
import json

def train_bpe(
    input_path: str | os.PathLike,  
    vocab_size: int,             
    special_tokens: list[str],   
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    #initialize token dict
    vocab = {i: bytes([i]) for i in range(256)}

    # iterate as long as it reaches the vocab capacity
    num_merges = vocab_size - 256 - len(special_tokens)

    # read input data, start training process
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if special_tokens:
        # use regex library to process special tokens
        # join together with "|" as splitting key
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        # split into segments
        parts = re.split(f"({special_regex})", text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    # Pre-tokenization process
    # Pre-tokenization rules used by GPT-2, separate out spaces, etc
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # Pretokenize each segment into GPT-2 words, then convert words into byte tuples, then count frequencies.
    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
    
    # Some preparation work for algorithm implementation

    # now we have a counter for words(dictionary of occunances)
    # word_list, count_list: 储存counter的keys，values
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple)) 
        counts_list.append(freq)

    # stats: store frequencies for byte pairs
    stats = defaultdict(int)

    # indices: stores are the indices of words that contain certain byte pair
    indices = defaultdict(set)


