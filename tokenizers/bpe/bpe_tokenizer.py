from tokenizers.base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def train(self):
        ...

    def encode(self):
        ...

    def decode(self):
        ...

    def save(self):
        ...

    def load(self):
        ...
