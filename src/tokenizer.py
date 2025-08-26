# tokenizer.py
"""
Minimal Byte-Pair Encoding (BPE) tokenizer implementation.
This tokenizer learns subword units from raw text data and can then encode text
into sequences of integer IDs suitable for feeding into neural networks.
"""

import json

def _get_stats(tokens_list):
    """
    Count frequency of all adjacent token pairs in a list of tokenized texts.

    Args:
        tokens_list (list[list[str]]): List of token sequences.

    Returns:
        dict: Mapping (token_a, token_b) -> count
    """
    stats = {}
    for toks in tokens_list:
        for i in range(len(toks)-1):
            pair = (toks[i], toks[i+1])
            stats[pair] = stats.get(pair, 0) + 1
    return stats


def _merge(tokens_list, pair, new_tok):
    """
    Merge a given token pair into a new token across all sequences.

    Args:
        tokens_list (list[list[str]]): Current tokenized sequences.
        pair (tuple): Pair of tokens (a, b) to merge.
        new_tok (str): New token formed from a+b.

    Returns:
        list[list[str]]: Updated tokenized sequences.
    """
    a, b = pair
    out = []
    for toks in tokens_list:
        i, merged = 0, []
        while i < len(toks):
            if i < len(toks)-1 and toks[i]==a and toks[i+1]==b:
                merged.append(new_tok); i += 2
            else:
                merged.append(toks[i]); i += 1
        out.append(merged)
    return out


class SimpleBPE:
    """
    A simple character-level Byte-Pair Encoding tokenizer.

    Attributes:
        vocab_size (int): Maximum vocabulary size.
        lowercase (bool): Whether to lowercase input text.
        special (list): Special tokens [PAD, SOS, EOS, UNK].
        id2tok (list): Vocabulary list mapping ID -> token.
        tok2id (dict): Reverse map token -> ID.
        merges (list): Sequence of learned merges (pairs of tokens).
    """
    def __init__(self, vocab_size=2000, lowercase=True):
        """
        Initialize SimpleBPE tokenizer.

        Args:
            vocab_size (int): Target vocabulary size.
            lowercase (bool): If True, lowercase all input text.
        """
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special = ["[PAD]","[SOS]","[EOS]","[UNK]"]
        self.id2tok = list(self.special)
        self.tok2id = {t:i for i,t in enumerate(self.id2tok)}
        self.merges = []

    def _init_base(self, texts):
        """
        Initialize vocabulary with characters present in the training texts.

        Args:
            texts (list[str]): Training corpus.
        """
        charset = set()
        for t in texts:
            if self.lowercase: t = t.lower()
            charset.update(list(t))
        base = sorted(list(charset)) + ["</w>"]  # end-of-word marker
        for tok in base:
            if tok not in self.tok2id:
                self.tok2id[tok] = len(self.id2tok)
                self.id2tok.append(tok)

    def _tokenize_chars(self, text):
        """
        Tokenize a text into characters + end-of-word marker.

        Args:
            text (str): Input text.

        Returns:
            list[str]: Token sequence.
        """
        if self.lowercase: text = text.lower()
        return list(text) + ["</w>"]

    def train(self, texts):
        """
        Train BPE tokenizer on given texts.

        Args:
            texts (list[str]): Training corpus.

        Updates:
            self.id2tok, self.tok2id, self.merges
        """
        self.id2tok = list(self.special)
        self.tok2id = {t:i for i,t in enumerate(self.id2tok)}
        self.merges = []
        self._init_base(texts)
        tokens_list = [self._tokenize_chars(t) for t in texts]

        while len(self.id2tok) < self.vocab_size:
            stats = _get_stats(tokens_list)
            if not stats: break
            (a,b), _ = max(stats.items(), key=lambda kv: kv[1])
            new_tok = a + b
            if new_tok in self.tok2id: break
            tokens_list = _merge(tokens_list, (a,b), new_tok)
            self.merges.append((a,b))
            self.tok2id[new_tok] = len(self.id2tok)
            self.id2tok.append(new_tok)

    def encode(self, text, max_len=77):
        """
        Encode text into a fixed-length sequence of token IDs.

        Args:
            text (str): Input string.
            max_len (int): Maximum output length.

        Returns:
            list[int]: Token IDs padded/truncated to max_len.
        """
        if self.lowercase: text = text.lower()
        toks = list(text) + ["</w>"]

        # Apply merges in the learned order
        for (a,b) in self.merges:
            i, merged = 0, []
            while i < len(toks):
                if i < len(toks)-1 and toks[i]==a and toks[i+1]==b:
                    merged.append(a+b); i += 2
                else:
                    merged.append(toks[i]); i += 1
            toks = merged

        ids = [self.tok2id.get(t, self.tok2id["[UNK]"]) for t in toks]
        ids = [self.tok2id["[SOS]"]] + ids + [self.tok2id["[EOS]"]]

        if len(ids) < max_len:
            ids += [self.tok2id["[PAD]"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.tok2id["[EOS]"]
        return ids

    def save(self, path):
        """
        Save tokenizer vocabulary + merges to JSON.

        Args:
            path (str): File path.
        """
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "lowercase": self.lowercase,
                "id2tok": self.id2tok,
                "merges": self.merges
            }, f)

    @classmethod
    def load(cls, path):
        """
        Load tokenizer from a saved JSON.

        Args:
            path (str): File path.

        Returns:
            SimpleBPE: Loaded tokenizer.
        """
        with open(path, "r") as f:
            obj = json.load(f)
        b = cls(vocab_size=obj["vocab_size"], lowercase=obj["lowercase"])
        b.id2tok = obj["id2tok"]
        b.tok2id = {t:i for i,t in enumerate(b.id2tok)}
        b.merges = [tuple(p) for p in obj["merges"]]
        return b
