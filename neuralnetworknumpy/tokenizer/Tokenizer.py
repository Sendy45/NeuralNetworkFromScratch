import re

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text):
        data = text.lower()
        data = self.pattern.findall(data)
        return data

    def encode(self, text):
        tokens = self.tokenize(text)

        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        return [self.vocab.token_to_id(tok) for tok in tokens]

    def decode(self, indices):
        tokens = [self.vocab.id_to_token(i) for i in indices]

        # strip special tokens before rendering
        tokens = [t for t in tokens if t not in (
            self.vocab.start_token, self.vocab.end_token,
            self.vocab.pad_token, "<CLS>", "<SEP>"
        )]

        text = ""
        for tok in tokens:
            if tok in {".", ",", "!", "?", ":", ";", "'"}:
                text += tok
            else:
                text += (" " if text else "") + tok
        return text.strip()

