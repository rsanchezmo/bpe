from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def encode(self):
        ...

    @abstractmethod
    def decode(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self):
        ...
