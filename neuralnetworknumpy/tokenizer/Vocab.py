import json

class Vocab:
    def __init__(self):
        self.vocab = []
        self.stoi = {}
        self.itos = {}

        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"


    # Build vocab for model from dataset
    def build(self, dataset):
        from .Tokenizer import Tokenizer
        tokenizer = Tokenizer(self)

        tokens = tokenizer.tokenize(dataset)

        unique_tokens = sorted(set(tokens))

        special_tokens = [self.pad_token, self.unk_token, "<CLS>", "<SEP>", self.start_token, self.end_token]

        self.vocab = special_tokens + [t for t in unique_tokens if t not in special_tokens]

        self.stoi = {tok: i for i, tok in enumerate(self.vocab)} # string to index
        self.itos = {i: tok for tok, i in self.stoi.items()} # index to string


    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)


    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)

        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab = list(self.stoi.keys())


    # String to Index
    def token_to_id(self, token):
        return self.stoi.get(token, self.stoi[self.unk_token])


    # Index to String
    def id_to_token(self, index):
        return self.itos[index]