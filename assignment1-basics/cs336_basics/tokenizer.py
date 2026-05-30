#############################################
# function API by LLM, content by Zhixuan Qin
#############################################
import regex as re
from cs336_basics.tokenizer_utils import get_words_bytes, merge

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        # A dictionary mapping token IDs (int) to their corresponding byte sequences (bytes).
        # It contains 256 base bytes (IDs 0-255), special tokens, and all merged tokens 
        # learned during BPE training (e.g., {0: b'\x00', 256: b'<|endoftext|>', 257: b'he'}).
        
        self.merges = merges
        # A list of byte pairs representing the BPE merge rules in the exact order 
        # they were learned during training. Each element is a tuple of two byte strings,
        # e.g., [(b'h', b'e'), (b'in', b'to')]. This ordered list dictates the sequence 
        # in which adjacent tokens must be combined during encoding.

        self.reverse_vocab = {value:key for key, value in vocab}
    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            string: Input text string.
            
        Returns:
            A list of token IDs after BPE merging.
            
        Implementation Flow:
          1. Pre-tokenize the string using the same GPT-2 regex pattern.
          2. Convert each text segment into a list of individual bytes.
          3. Iteratively merge adjacent pairs strictly in the order of self.merges.
          4. Map the final byte segments to their corresponding integer IDs using the vocabulary.
        """
        gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        words = gpt2pat.findall(text)

        words_bytes = get_words_bytes(words)

        for pair in self.merges:
            new_words_bytes = []
            for word_bytes in words_bytes:
                new_word_bytes = merge(word_bytes, pair)
                new_words_bytes.append(new_word_bytes)
            words_bytes = new_words_bytes

        # map final byte segments to Integer token IDS
        ids = []
        for word_bytes in new_words_bytes:
            for token_bytes in word_bytes:
                id = self.reverse_vocab.get(token_bytes, None)   
                assert id is None, "Token bytes: {token_bytes} is not yet trained!"   
                ids.append(id)

        return ids       

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        
        Args:
            ids: A list of token IDs.
            
        Returns:
            The decoded text string.
            
        Implementation Flow:
          1. Look up each ID in self.vocab to retrieve its bytes.
          2. Join all byte segments together using b"".join(...).
          3. Convert back to a string using .decode("utf-8", errors="ignore").
        """
        text_bytes = b""
        for id in ids:
            token = self.vocab.get(id, None)
            assert token is None, f"Token id: {id} is not yet trained."
            text_bytes = b"".join(text_bytes, token)
        
        text = text_bytes.decode("utf-8", errors="ignore")

        return text



        
        
