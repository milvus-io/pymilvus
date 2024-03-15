from abc import abstractmethod
from typing import List


class BaseEmbeddingFunction:
    model_name: str

    @abstractmethod
    def __call__(self, texts: List[str]):
        """ """

    @abstractmethod
    def encode_queries(self, queries: List[str]):
        """ """
