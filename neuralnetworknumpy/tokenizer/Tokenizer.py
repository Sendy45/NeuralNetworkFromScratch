import json
import regex as re

# Regex pattern used by gpt4 to split text into words for tokenizing
# handles 'll, 've, 's, 'd, 'm, 't, 're, words, numbers up to 3 digits long, spaces
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self):

        # merges - bank of merged tokens - used for encoding
        self.merges = {}  # (int, int) -> int

        # Bank of special tokens, goes both ways
        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}  # int -> str

        # vocab - all tokens and ids - used for decoding
        self.vocab = self._build_vocab()  # int -> bytes

        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Create vocab from ids and merged tokens
    def _build_vocab(self):
        # First 256 ids
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Merges handling
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # Special tokens handling
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode("utf-8")
        return vocab

    # Get counter of pairs
    @staticmethod
    def _get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # Iterate over consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # Merge a pair - give them a new token idx
    @staticmethod
    def _merge(ids, pair, new_idx):
        new_ids = []
        i = 0
        # Go over ids and replace the pair with the new idx
        while i < len(ids):
            # Pair found, replace and move 2 up
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # Train tokenizer from text
    # vocab_size - max number of tokens for the tokenizer
    # verbose - print merges
    def fit(self, text, vocab_size, verbose=False):

        # Min vocab_size = 256
        assert vocab_size >= 256

        # num of merges = num until desired vocab_size
        num_merges = vocab_size - 256

        # Split into chunks - merges never cross chunk boundaries
        chunks = re.findall(self.compiled_pattern, text)

        # Encode the text chunks
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        self.merges = {}

        # Merge until hitting max vocab_size
        for i in range(num_merges):

            # Get stats (pairs) for each chunk
            stats = {}
            for chuck_ids in ids:
                self._get_stats(chuck_ids, stats)

            if not stats:  # Empty
                break

            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            # Give it a new token idx
            idx = 256 + i
            # Merge for all chunks
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # Update merges
            self.merges[pair] = idx

            # Print for debugging
            if verbose:
                merged_bytes = self._build_vocab().get(idx, b"?")
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({merged_bytes}) count={stats[pair]}")

        # Build vocab
        self.vocab = self._build_vocab()

        return ids

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    # Encode chunk - handle merges
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes) # create a copy
        while len(ids) >= 2: # if len(ids)==1 -> no pairs
            # Get all pairs
            stats = self._get_stats(ids)
            # find the pair with the lowest merge index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # Got to unknown pair or float("inf") -> break
            if pair not in self.merges:
                break
            # Merge pairs
            ids = self._merge(ids, pair, self.merges[pair])
        return ids


    # Encode without handling special tokens
    def encode_ordinary(self, text):
        chunks = re.findall(self.compiled_pattern, text) # apply regex pattern
        ids = []
        for chunk in chunks:
            # Encode ( to utf-8 and merges )
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    # Main encode function - handle chunks and special tokens
    def encode(self, text, allowed_special="all"):
        """
        Encode text, optionally handling special tokens.
        allowed_special: "all" | "none" | set of allowed special token strings
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # If no special tokens, encode regularly
            return self.encode_ordinary(text)

        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")" # Regex pattern
        chunks = re.split(special_pattern, text)

        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in chunks:
            if chunk in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[chunk])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(chunk))
        return ids

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, ids):
        parts = [] # Construct all parts
        for idx in ids:
            # Reconstruct bytes
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            # handle special tokens (using inverse_special_tokens)
            elif idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else: # Unrecognizable idx
                # todo swap to <UNK>
                raise ValueError(f"Unknown token id: {idx}")
        # Build string, handle problematic bytes ids by replacing them
        return b"".join(parts).decode("utf-8", errors="replace")

    # -------------------
    # Saving / Loading
    # -------------------
    def save(self, path):
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},  # bytes → list of ints
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}  # list of ints → bytes
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}