# hello_model.py simplifies the demonstration of using various embedding functions in PyMilvus,
# focusing on dense, sparse, and hybrid models. This script illustrates:
# - Initializing and using OpenAIEmbeddingFunction for dense embeddings
# - Initializing and using BGEM3EmbeddingFunction for hybrid embeddings
# - Initializing and using SentenceTransformerEmbeddingFunction for dense embeddings
# - Initializing and using BM25EmbeddingFunction for sparse embeddings
# - Initializing and using SpladeEmbeddingFunction for sparse embeddings
import time

from pymilvus.model.dense import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.sparse import BM25EmbeddingFunction, SpladeEmbeddingFunction

fmt = "=== {:30} ==="


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + msg)


# OpenAIEmbeddingFunction usage
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
log(fmt.format("OpenAIEmbeddingFunction Usage"))

ef_openai = OpenAIEmbeddingFunction(api_key="sk-your-api-key")
embs_openai = ef_openai(docs)
log(f"Dimension: {ef_openai.dim} Embedding Shape: {embs_openai[0].shape}")

# -----------------------------------------------------------------------------
# BGEM3EmbeddingFunction usage
log(fmt.format("BGEM3EmbeddingFunction Usage"))
ef_bge = BGEM3EmbeddingFunction(device="cpu", use_fp16=False)
embs_bge = ef_bge(docs)
log("Embedding Shape: {} Dimension: {}".format(embs_bge["dense"][0].shape, ef_bge.dim))

# -----------------------------------------------------------------------------
# SentenceTransformerEmbeddingFunction usage
log(fmt.format("SentenceTransformerEmbeddingFunction Usage"))
ef_sentence_transformer = SentenceTransformerEmbeddingFunction(device="cpu")
embs_sentence_transformer = ef_sentence_transformer(docs)
log(
    "Embedding Shape: {} Dimension: {}".format(
        embs_sentence_transformer[0].shape, ef_sentence_transformer.dim
    )
)

# -----------------------------------------------------------------------------
# BM25EmbeddingFunction usage
log(fmt.format("BM25EmbeddingFunction Usage"))
ef_bm25 = BM25EmbeddingFunction()
ef_bm25.load()
embs_bm25 = ef_bm25.encode_documents(docs)
log(f"Embedding Shape: {embs_bm25[:, [0]].shape} Dimension: {ef_bm25.dim}")

# -----------------------------------------------------------------------------
# SpladeEmbeddingFunction usage
log(fmt.format("SpladeEmbeddingFunction Usage"))
ef_splade = SpladeEmbeddingFunction(device="cpu")
embs_splade = ef_splade(["Hello world", "Hello world2"])
log(f"Embedding Shape: {embs_splade[:, [0]].shape} Dimension: {ef_splade.dim}")

# -----------------------------------------------------------------------------
log(fmt.format("Demonstrations Finished"))
