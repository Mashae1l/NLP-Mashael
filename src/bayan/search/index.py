"""Build and persist Bayan's versioned FAISS case index."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DEFAULT_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_CASES_PATH = Path(
    "data/search/bayan_cases.csv"
)


def build_index(
    prefix: str,
    limit: int | None = None,
    model_name: str = DEFAULT_MODEL,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    batch_size: int = 128,
) -> dict:
    """Encode cases, L2-normalise vectors, and persist index files."""

    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = Path(f"{prefix_path}.faiss")
    metadata_path = Path(
        f"{prefix_path}_metadata.jsonl"
    )
    manifest_path = Path(
        f"{prefix_path}_manifest.json"
    )

    cases = pd.read_csv(cases_path)

    required_columns = {
        "case_id",
        "lang",
        "topic",
        "case_text",
        "resolution",
    }

    missing_columns = required_columns.difference(
        cases.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing case columns: "
            f"{sorted(missing_columns)}"
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        cases = cases.head(limit).copy()

    if cases.empty:
        raise ValueError(
            "Cannot build an index from an empty corpus"
        )

    cases = cases.reset_index(drop=True)

    cases["indexed_text"] = cases[
        "case_text"
    ].fillna("").map(preprocess)

    model = SentenceTransformer(model_name)

    vectors = model.encode(
        cases["indexed_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=len(cases) > 100,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
        order="C",
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Encoder must return a 2D vector matrix"
        )

    faiss.normalize_L2(vectors)

    dimension = int(vectors.shape[1])

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    faiss.write_index(
        index,
        str(index_path),
    )

    metadata_columns = [
        "case_id",
        "lang",
        "topic",
        "case_text",
        "resolution",
        "status",
        "closed_at",
        "indexed_text",
    ]

    available_columns = [
        column
        for column in metadata_columns
        if column in cases.columns
    ]

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        for record in cases[
            available_columns
        ].to_dict(orient="records"):
            metadata_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": dimension,
        "metric": "inner_product",
        "l2_normalised": True,
        "index_path": index_path.name,
        "metadata_path": metadata_path.name,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("FAISS index saved to:", index_path)
    print("Metadata saved to:", metadata_path)
    print("Manifest saved to:", manifest_path)
    print("Vectors:", manifest["n_vectors"])
    print("Dimension:", manifest["dim"])

    return manifest
