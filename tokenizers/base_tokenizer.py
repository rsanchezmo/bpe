from abc import abstractmethod
from typing import List
from pathlib import Path


class BaseTokenizer:
    def __init__(self):
        self.vocab_size = 0

    @abstractmethod
    def train(self, dataset: str, vocab_size: int):
        ...

    @abstractmethod
    def encode(self, txt: str):
        ...

    @abstractmethod
    def decode(self, encoded: List[int]) -> str:
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self, path: Path):
        ...
