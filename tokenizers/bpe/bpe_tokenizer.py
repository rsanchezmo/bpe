from tokenizers.base_tokenizer import BaseTokenizer
from tokenizers.bpe.utils import get_consecutive_pairs, encode_utf8

from typing import List


class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.merges = {}

    def train(self, dataset: str, vocab_size: int):
        """Trains the tokenizer using the BPE algorithm.
        The dataset is a string with the text to train the tokenizer.
        The vocab_size is the maximum number of tokens to generate.
        Returns a list of integers, where each integer is a token.
        The self.merges dictionary contains the pairs of bytes that were merged and the value assigned to them."""

        encoded_list = encode_utf8(dataset)

        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 due to the UTF-8 encoding")

        for vocab_size_value in range(256, vocab_size+1):
            pairs = get_consecutive_pairs(encoded_list)
            max_pair = max(pairs, key=pairs.get)

            if pairs[max_pair] == 1:
                # no more pairs to merge
                break

            new_list = []
            i = 0
            while i < len(encoded_list):
                if i < len(encoded_list) - 1 and encoded_list[i] == max_pair[0] and encoded_list[i+1] == max_pair[1]:
                    self.merges[max_pair] = vocab_size_value
                    new_list.append(vocab_size_value)
                    i += 1
                else:
                    new_list.append(encoded_list[i])
                i += 1

            encoded_list = new_list

        self.vocab_size = vocab_size_value + 1

        return encoded_list

    def encode(self, txt: str) -> List[int]:
        """Encodes a string using the merges learned during training, and returns a list of integers.
        If a pair of bytes is found in the merges, it is replaced by the value in the self.merges dictionary.
        If the dict is empty, the list of integers is the UTF-8 encoding of the string."""

        utf8_encoded = encode_utf8(txt)
        for key, value in self.merges.items():
            i = 0
            new_list = []
            while i < len(utf8_encoded):
                if i < len(utf8_encoded) - 1 and utf8_encoded[i] == key[0] and utf8_encoded[i+1] == key[1]:
                    new_list.append(value)
                    i += 1
                else:
                    new_list.append(utf8_encoded[i])
                i += 1

            utf8_encoded = new_list

        return utf8_encoded

    def decode(self, encoded: List[int]) -> str:

        # decode from last to first
        for key, value in reversed(self.merges.items()):
            decoded = []
            i = 0
            while i < len(encoded):
                if i < len(encoded) - 1 and encoded[i] == value:
                    decoded.append(key[0])
                    decoded.append(key[1])
                else:
                    decoded.append(encoded[i])
                i += 1
            encoded = decoded

        return bytes(decoded).decode('utf-8', errors='replace')

    def save(self):
        ...

    def load(self):
        ...


if __name__ == '__main__':
    tokenizer = BPETokenizer()
    txt = "hola me llamo rodrigo y tu? ### ### for i in range():"
    print('UTF-8 encoding size:', len(txt.encode('utf-8')))
    tokenizer.train(txt, 300)
    print(tokenizer.merges)
    print(f'Vocab size: {tokenizer.vocab_size}')
    encoded = tokenizer.encode(txt)
    print(encoded)
    print('Tokens size:', len(encoded))
    decoded = tokenizer.decode(encoded)
    print(decoded)
    print(txt == decoded)
