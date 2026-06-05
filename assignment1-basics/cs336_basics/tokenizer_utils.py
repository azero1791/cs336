import os
import io
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries

def train_bpe_logic(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Core BPE training control flow.
    """
    # 1. Initialize the vocabulary with 256 base bytes and special_tokens.
    #    Assign initial token IDs accordingly.
    
    # 2. Call bpe_utils.initialize_word_counts to get the initial word count dict.
    
    # 3. Core BPE loop:
    #    while len(current_vocab) < vocab_size:
    #        a. Call get_pair_counts to count current pair frequencies.
    #        b. Find the most frequent pair (break if max frequency is 1 or no pairs left).
    #        c. Append the best_pair to the merges list.
    #        d. Concatenate the best_pair to form a new token, assign a new ID, and add to vocab.
    #        e. Call merge_word_counts to update the global word count dict for the next iteration.
    
    # 4. Return the final (vocab, merges) tuple.
    vocab = {}

    # initial vocab with 256 base bytes and special_tokens
    for i in range(256):
        vocab[i] = bytes([i])
    
    id = 256
    for special_token in special_tokens:
        special_token_bytes = special_token.encode("utf-8")
        vocab[id] = special_token_bytes
        id += 1

    # get initial words counts
    words_counts = initialize_word_counts(input_path, special_tokens, 4)

    merges_list = []
    while len(vocab) < vocab_size:
        pairs = get_pair_counts(words_counts)

        max_item = max(pairs.items(), key=lambda item: (item[1], item[0]), default=None)
        if max_item is None or max_item[1] == 1:
            break   
        max_frequency_pair = max_item[0]

        merges_list.append(max_frequency_pair)
        
        merged_bytes = b"".join(max_frequency_pair)
        vocab[id] = merged_bytes
        id += 1

        words_counts = merge_word_counts(words_counts, max_frequency_pair)

    return vocab, merges_list

# reference: https://github.com/weiruihhh/cs336_note_and_hw/blob/main/chapter1/hw1/pair_all_bpe_tokenzier.py
def initialize_word_counts(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_workers: int = 1
) -> dict[tuple[bytes, ...], int]:
    """
    Step 1: Pre-tokenization and frequency initialization.
    
    Args:
        input_path: Path to the training corpus file.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        num_workers: Number of processes for parallel chunk reading.
        
    Returns:
        A global word count dictionary.
        Key: A tuple of individual bytes, e.g., (b' ', b'w', b'o', b'r', b'l', b'd')
        Value: The total frequency of this pre-token after regex splitting.
    """

    try:
        with open(input_path, "rb") as f_disk:
            disk_bytes = f_disk.read()
        #  windows: \r\n -> linux: \n
        linux_bytes = disk_bytes.replace(b"\r\n", b"\n")
    except FileNotFoundError:
        return {}

    text = linux_bytes.decode("utf-8", errors="ignore")

    split_pattern = '|'.join(map(re.escape, special_tokens))
    
    if split_pattern:
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    count_dic = {}
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    for chunk in chunks:
        for word in re.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            
            byte_tuple = tuple(bytes([b]) for b in word_bytes)
            
            count_dic[byte_tuple] = count_dic.get(byte_tuple, 0) + 1

    return count_dic

def get_pair_counts(
    word_counts: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    """
    Step 2: Count all adjacent byte pairs in the current vocabulary.
    
    Args:
        word_counts: The current global word count dictionary.
        
    Returns:
        A dictionary of adjacent pair frequencies.
        Key: A tuple of two adjacent elements, e.g., (b'l', b'o')
        Value: Total occurrences in the corpus (multiplied by the word's frequency).
    """
    pair_dic = {}
    for word_bytes, frequency in word_counts.items():
        # word_bytes is special_token byte or only one byte
        if len(word_bytes) == 1:
            continue
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_dic[pair] = pair_dic.get(pair, 0) + frequency
        
    return pair_dic
        

def merge_word_counts(
    word_counts: dict[tuple[bytes, ...], int], 
    best_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    """
    Step 3: Globally replace and merge the most frequent byte pair.
    
    Args:
        word_counts: The old global word count dictionary.
        best_pair: The most frequent pair chosen for this iteration, e.g., (b'h', b'e')
        
    Returns:
        A new global word count dictionary where all consecutive occurrences of 
        best_pair are merged into a single atomic byte string element (b'he').
        E.g., Key changes from (b'h', b'e', b'l', b'l', b'o') to (b'he', b'l', b'l', b'o').
    """
    new_word_counts = {}
    for word_byte, frequency in word_counts.items():
        new_word_byte = merge(word_byte, best_pair)
        new_word_counts[new_word_byte] = new_word_counts.get(new_word_byte, 0) + frequency
                
    return new_word_counts

def get_words_bytes(words: list[str], special_tokens: list[str]):
    """
    Args:
        words: List of string word
    Returns:
        A list of words bytes
    """
    words_bytes = []
    for word in words:
        
        word_bytes = word.encode("utf-8")
        if special_tokens is not None and word in special_tokens:
            byte_tuple = tuple([word_bytes])
        else:
            byte_tuple = tuple(bytes([b]) for b in word_bytes)
        words_bytes.append(byte_tuple)
    
    return words_bytes

def merge(word_bytes: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Args:
        word_bytes: tuple of word bytes
        pair: word bytes to be merged into word_bytes
    Returns:
        A tuple of bytes
    """
    new_word_bytes = []
    i = 0
    word_bytes_lst = list(word_bytes)
    merged_bytes = b"".join(pair)
    while i < len(word_bytes_lst):
        if i == len(word_bytes_lst) - 1 or word_bytes_lst[i] != pair[0] or word_bytes_lst[i+1] != pair[1]:
            new_word_bytes.append(word_bytes_lst[i])
            i += 1
        else:
            new_word_bytes.append(merged_bytes)
            i += 2
    return tuple(new_word_bytes)

def special_aware_split(text: str, special_tokens : list[str]) -> list[str]:
    """
    Args:
        text: raw text may containing special tokens to be extracted
        special_tokens: a list of special tokens 
    Returns:
        a list of splited text containing nonspecial text and special text 
    """
    text_lst = []
    pre = 0
    cur = 0
    if special_tokens is None:
        text_lst.append(text)
        return text_lst
    
    while cur < len(text):
        flag = False
        for special_token in special_tokens:
            if (text.startswith(special_token, cur)):
                if pre != cur:
                    text_lst.append(text[pre:cur])
                text_lst.append(special_token)
                cur += len(special_token)
                pre = cur
                flag = True
                break
        if flag == False:
            cur += 1
    
    if pre <= cur - 1:
        text_lst.append(text[pre: cur])
    
    return text_lst
            