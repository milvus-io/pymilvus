"""
This file incorporates components from the 'rank_bm25' project by Dorian Brown:
https://github.com/dorianbrown/rank_bm25
Specifically, the rank_bm25.py file.

The incorporated components are licensed under the Apache License, Version 2.0 (the "License");
you may not use these components except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

import requests
from scipy.sparse import csr_array, vstack

from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import Analyzer, build_default_analyzer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


class BM25EmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        analyzer: Analyzer = None,
        corpus: Optional[List] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        num_workers: Optional[int] = None,
    ):
        if analyzer is None:
            analyzer = build_default_analyzer(language="en")
        self.analyzer = analyzer
        self.corpus_size = 0
        self.avgdl = 0
        self.idf = {}
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        if num_workers is None:
            self.num_workers = cpu_count()
        self.num_workers = num_workers

        if analyzer and corpus is not None:
            self.fit(corpus)

    def _calc_term_indices(self):
        for index, word in enumerate(self.idf):
            self.idf[word][1] = index

    def _compute_statistics(self, corpus: List[str]):
        term_document_frequencies = defaultdict(int)
        total_word_count = 0
        for document in corpus:
            total_word_count += len(document)

            frequencies = defaultdict(int)
            for word in document:
                frequencies[word] += 1

            for word, _ in frequencies.items():
                term_document_frequencies[word] += 1
            self.corpus_size += 1
        self.avgdl = total_word_count / self.corpus_size
        return term_document_frequencies

    def _tokenize_corpus(self, corpus: List[str]):
        if self.num_workers == 1:
            return [self.analyzer(text) for text in corpus]
        pool = Pool(self.num_workers)
        return pool.map(self.analyzer, corpus)

    def _calc_idf(self, term_document_frequencies: Dict):
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in term_document_frequencies.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            if word not in self.idf:
                self.idf[word] = [0.0, 0]
            self.idf[word][0] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word][0] = eps

    def _rebuild(self, corpus: List[str]):
        self._clear()
        corpus = self._tokenize_corpus(corpus)
        term_document_frequencies = self._compute_statistics(corpus)
        self._calc_idf(term_document_frequencies)
        self._calc_term_indices()

    def _clear(self):
        self.corpus_size = 0
        # idf records the (value, index)
        self.idf = defaultdict(list)

    @property
    def dim(self):
        return len(self.idf)

    def fit(self, corpus: List[str]):
        self._rebuild(corpus)

    def _encode_query(self, query: str) -> csr_array:
        terms = self.analyzer(query)
        values, rows, cols = [], [], []
        for term in terms:
            if term in self.idf:
                values.append(self.idf[term][0])
                rows.append(0)
                cols.append(self.idf[term][1])
        return csr_array((values, (rows, cols)), shape=(1, len(self.idf)))

    def _encode_document(self, doc: str) -> csr_array:
        terms = self.analyzer(doc)
        frequencies = defaultdict(int)
        doc_len = len(terms)
        term_set = set()
        for term in terms:
            frequencies[term] += 1
            term_set.add(term)
        values, rows, cols = [], [], []
        for term in term_set:
            if term in self.idf:
                term_freq = frequencies[term]
                value = (
                    term_freq
                    * (self.k1 + 1)
                    / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                )
                rows.append(0)
                cols.append(self.idf[term][1])
                values.append(value)
        return csr_array((values, (rows, cols)), shape=(1, len(self.idf)))

    def encode_queries(self, queries: List[str]) -> csr_array:
        sparse_embs = [self._encode_query(query) for query in queries]
        return vstack(sparse_embs)

    def __call__(self, texts: List[str]) -> csr_array:
        error_message = "Unsupported function called, please check the documentation of 'BM25EmbeddingFunction'."
        raise ValueError(error_message)

    def encode_documents(self, documents: List[str]) -> csr_array:
        sparse_embs = [self._encode_document(document) for document in documents]
        return vstack(sparse_embs)

    def save(self, path: str):
        bm25_params = {}
        bm25_params["version"] = "v1"
        bm25_params["corpus_size"] = self.corpus_size
        bm25_params["avgdl"] = self.avgdl
        bm25_params["idf_word"] = [None for _ in range(len(self.idf))]
        bm25_params["idf_value"] = [None for _ in range(len(self.idf))]
        for word, values in self.idf.items():
            bm25_params["idf_word"][values[1]] = word
            bm25_params["idf_value"][values[1]] = values[0]

        bm25_params["k1"] = self.k1
        bm25_params["b"] = self.b
        bm25_params["epsilon"] = self.epsilon

        with Path(path).open("w") as json_file:
            json.dump(bm25_params, json_file)

    def load(self, path: Optional[str] = None):
        default_meta_filename = "bm25_msmarco_v1.json"
        default_meta_url = "https://github.com/milvus-io/pymilvus-assets/releases/download/v0.1-bm25v1/bm25_msmarco_v1.json"
        if path is None:
            logger.info(f"path is None, using default {default_meta_filename}.")
            if not Path(default_meta_filename).exists():
                try:
                    logger.info(
                        f"{default_meta_filename} not found, start downloading from {default_meta_url} to ./{default_meta_filename}."
                    )
                    response = requests.get(default_meta_url, timeout=30)
                    response.raise_for_status()
                    with Path(default_meta_filename).open("wb") as f:
                        f.write(response.content)
                    logger.info(f"{default_meta_filename} has been downloaded successfully.")
                except requests.exceptions.RequestException as e:
                    error_message = f"Failed to download the file: {e}"
                    raise RuntimeError(error_message) from e
            path = default_meta_filename
        try:
            with Path(path).open() as json_file:
                bm25_params = json.load(json_file)
        except OSError as e:
            error_message = f"Error opening file {path}: {e}"
            raise RuntimeError(error_message) from e
        self.corpus_size = bm25_params["corpus_size"]
        self.avgdl = bm25_params["avgdl"]
        self.idf = {}
        for i in range(len(bm25_params["idf_word"])):
            self.idf[bm25_params["idf_word"][i]] = [bm25_params["idf_value"][i], i]
        self.k1 = bm25_params["k1"]
        self.b = bm25_params["b"]
        self.epsilon = bm25_params["epsilon"]
