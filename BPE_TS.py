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

    # initialize stats and indices
    for idx, word in enumerate(words_list):
        freq = counts_list[idx] 
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq         
            indices[pair].add(idx)      

    merges = []
    for _ in range(num_merges):
        if not stats:
            break
        
        # Step 1: find the best pair, pair with most frequency
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        
        if stats[best_pair] <= 0:
            break
        
        # Step 2: access all indices need to be updated
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        relevant_indices = list(indices[best_pair])
        
        for idx in relevant_indices:
            word = words_list[idx] 
            freq = counts_list[idx] 
            
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                      
                    word[i] = new_token    
                    del word[i+1]          

                    if i > 0:
                        new_prev = (word[i-1], word[i]) 
                        stats[new_prev] += freq
                        indices[new_prev].add(idx) 
                    
                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)

                else:
                    i += 1

        # clean the merged byte pair 
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # Build the final word dictionary
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
        
    # Add in special tokens
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges