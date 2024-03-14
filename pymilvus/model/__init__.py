from . import dense, hybrid, sparse
from .dense.sentence_transformer import SentenceTransformerEmbeddingFunction

__all__ = ["DefaultEmbeddingFunction", "dense", "sparse", "hybrid"]

DefaultEmbeddingFunction = SentenceTransformerEmbeddingFunction
