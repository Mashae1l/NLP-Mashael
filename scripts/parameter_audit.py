"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

import gc

from transformers import AutoConfig, AutoModel


def audit(checkpoint: str) -> dict:
    """Count model parameters by component."""

    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModel.from_config(config)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        lower_name = name.lower()

        if "embeddings" in lower_name:
            buckets["embeddings"] += count

        elif "attention" in lower_name:
            buckets["attention"] += count

        elif "layernorm" in lower_name or "layer_norm" in lower_name:
            buckets["norms"] += count

        elif "intermediate" in lower_name or "output.dense" in lower_name:
            buckets["ffn"] += count

        elif "pooler" in lower_name:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    total = sum(buckets.values())

    result = {
        "total": total,
        **buckets,
        "embedding_share": buckets["embeddings"] / total * 100,
    }

    del model
    gc.collect()

    return result


def print_audit(checkpoint: str, result: dict):
    """Print the parameter audit results."""

    print(f"\nModel: {checkpoint}")
    print(f"Total parameters: {result['total']:,}")

    bucket_names = [
        "embeddings",
        "attention",
        "ffn",
        "norms",
        "pooler",
        "other",
    ]

    for bucket in bucket_names:
        count = result[bucket]
        percentage = count / result["total"] * 100

        print(
            f"{bucket:12s}: {count:12,d} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"Embedding share: "
        f"{result['embedding_share']:.2f}%"
    )


def main():
    checkpoints = [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]

    for checkpoint in checkpoints:
        print(f"\nLoading {checkpoint}...")

        result = audit(checkpoint)
        print_audit(checkpoint, result)

    print(
        "\nEmbedding-share explanation: "
        "mBERT has a larger multilingual vocabulary, so its embedding "
        "matrix consumes a larger share of the model parameters."
    )


if __name__ == "__main__":
    main()
