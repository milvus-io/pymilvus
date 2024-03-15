from collections import defaultdict
from typing import List, Optional

import numpy as np

from pymilvus.model.base import BaseEmbeddingFunction


class OpenAIEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as err:
            error_message = "openai is not installed."
            raise ImportError(error_message) from err

        self._openai_model_meta_info = defaultdict(dict)
        self._openai_model_meta_info["text-embedding-3-small"]["dim"] = 1536
        self._openai_model_meta_info["text-embedding-3-large"]["dim"] = 3072
        self._openai_model_meta_info["text-embedding-ada-002"]["dim"] = 1536

        self._model_config = dict({"api_key": api_key, "base_url": base_url}, **kwargs)
        additional_encode_config = {}
        if dimensions is not None:
            additional_encode_config = {"dimensions": dimensions}
            self._openai_model_meta_info[model_name]["dim"] = dimensions

        self._encode_config = {"model": model_name, **additional_encode_config}
        self.model_name = model_name
        self.client = OpenAI(**self._model_config)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    @property
    def dim(self):
        return self._openai_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode_query(self, query: str) -> np.array:
        return self._encode(query)[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode(document)[0]

    def _call_openai_api(self, texts: List[str]):
        results = self.client.embeddings.create(input=texts, **self._encode_config).data
        return [np.array(data.embedding) for data in results]

    def _encode(self, texts: List[str]):
        return self._call_openai_api(texts)
