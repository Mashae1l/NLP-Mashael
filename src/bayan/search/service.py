"""Two-stage bilingual semantic search for Bayan cases."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from bayan.preprocessing.core import (
    PREPROC_VERSION,
    preprocess,
)


DEFAULT_RERANKER = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


class CaseSearch:
    """Load a persisted index and perform retrieval plus reranking."""

    def __init__(
        self,
        prefix: str,
        reranker_name: str = DEFAULT_RERANKER,
    ):
        self.prefix = Path(prefix)

        manifest_path = Path(
            f"{self.prefix}_manifest.json"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing index manifest: {manifest_path}"
            )

        self.manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        required_keys = {
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
            "metric",
            "l2_normalised",
            "index_path",
            "metadata_path",
        }

        missing_keys = required_keys.difference(
            self.manifest
        )

        if missing_keys:
            raise ValueError(
                "Manifest is missing keys: "
                f"{sorted(missing_keys)}"
            )

        if (
            self.manifest["preproc_version"]
            != PREPROC_VERSION
        ):
            raise ValueError(
                "Index preprocessing version does not "
                "match the current pipeline"
            )

        if not self.manifest["l2_normalised"]:
            raise ValueError(
                "Bayan search requires an "
                "L2-normalised index"
            )

        index_path = (
            self.prefix.parent
            / self.manifest["index_path"]
        )

        metadata_path = (
            self.prefix.parent
            / self.manifest["metadata_path"]
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS index: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata: {metadata_path}"
            )

        self.index = faiss.read_index(
            str(index_path)
        )

        self.metadata = []

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            for line in metadata_file:
                if line.strip():
                    self.metadata.append(
                        json.loads(line)
                    )

        expected_vectors = int(
            self.manifest["n_vectors"]
        )

        expected_dimension = int(
            self.manifest["dim"]
        )

        if self.index.ntotal != expected_vectors:
            raise ValueError(
                "FAISS vector count does not match "
                "the manifest"
            )

        if self.index.d != expected_dimension:
            raise ValueError(
                "FAISS dimension does not match "
                "the manifest"
            )

        if len(self.metadata) != expected_vectors:
            raise ValueError(
                "Metadata count does not match "
                "the FAISS index"
            )

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            reranker_name
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
        rerank: bool = True,
    ) -> list[dict]:
        """Search cases and return ranked results."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if k <= 0 or candidates <= 0:
            raise ValueError(
                "k and candidates must be positive"
            )

        clean_query = preprocess(query)

        if not clean_query:
            return []

        query_vector = self.encoder.encode(
            [clean_query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
            order="C",
        )

        faiss.normalize_L2(query_vector)

        candidate_count = min(
            max(k, candidates),
            self.index.ntotal,
        )

        similarities, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        retrieved = []

        for similarity, index_position in zip(
            similarities[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            record = dict(
                self.metadata[int(index_position)]
            )

            record["bi_encoder_score"] = float(
                similarity
            )

            retrieved.append(record)

        if not retrieved:
            return []

        if rerank:
            pairs = [
                [
                    clean_query,
                    record["indexed_text"],
                ]
                for record in retrieved
            ]

            reranker_scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
            )

            reranker_scores = np.asarray(
                reranker_scores
            ).reshape(-1)

            for record, score in zip(
                retrieved,
                reranker_scores,
            ):
                record["score"] = float(score)

            retrieved.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

        else:
            for record in retrieved:
                record["score"] = record[
                    "bi_encoder_score"
                ]

            retrieved.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

        if retrieved[0]["score"] < min_score:
            return []

        results = retrieved[:k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rank"] = rank

        return results
