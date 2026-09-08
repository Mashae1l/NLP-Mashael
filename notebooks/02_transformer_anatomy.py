"""Lab 2: Transformer anatomy and attention verification."""

import math

import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


def calculate_weights(q, k, mask=None):
    """Calculate attention weights for inspection."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    return torch.softmax(scores, dim=-1)


def main():
    torch.manual_seed(42)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    # Compare our implementation with PyTorch
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    max_difference = (actual - expected).abs().max().item()
    equivalent = torch.allclose(actual, expected, atol=1e-6)

    print("Numerical equivalence:", equivalent)
    print("Maximum difference:", max_difference)

    assert equivalent, "Attention does not match PyTorch"

    # Inspect one attention head
    weights = calculate_weights(q, k)

    print("\nAttention weight matrix — first head:")
    print(weights[0, 0])

    # Exercise Multi-Head Attention
    model = MultiHeadAttention(d_model=8, num_heads=2)
    x = torch.randn(1, 4, 8)

    mha_output = model(x, x, x)

    print("\nMHA output shape:", tuple(mha_output.shape))
    assert mha_output.shape == x.shape

    # Verify a padding-style mask
    key_mask = torch.tensor(
        [[[[True, True, True, False]]]]
    )

    masked_weights = calculate_weights(q, k, key_mask)

    print("\nMasked final-token attention:")
    print(masked_weights[0, 0, :, -1])

    assert torch.all(masked_weights[..., -1] == 0)
    # Step 4: Causal mask
    sequence_length = q.size(-2)

    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        )
    ).view(1, 1, sequence_length, sequence_length)

    causal_weights = calculate_weights(q, k, causal_mask)

    future_mass = torch.triu(
        causal_weights,
        diagonal=1,
    ).sum().item()

    lower_triangular = torch.all(
        causal_weights == torch.tril(causal_weights)
    ).item()

    print("\nCausal attention matrix:")
    print(causal_weights[0, 0])
    print("Lower triangular:", lower_triangular)
    print("Future attention mass:", future_mass)
    print("Model family: Decoder-style causal attention")

    assert lower_triangular
    assert future_mass == 0.0

    # Step 5: Attention-map and padding diagnostics
    diagnostic_tokens = [
        "[CLS]",
        "الخدمة",
        "[SEP]",
        "[PAD]",
    ]

    adjacency_mass = torch.diagonal(
        weights[0, 0],
        offset=1,
    ).mean().item()

    sep_sink_mass = weights[..., 2].mean().item()
    pad_mass_without_mask = weights[..., -1].mean().item()
    pad_mass_with_mask = masked_weights[..., -1].mean().item()

    print("\nDiagnostic tokens:", diagnostic_tokens)
    print("Adjacent-token attention mass:", adjacency_mass)
    print("[SEP] sink mass:", sep_sink_mass)
    print("Pad mass without mask:", pad_mass_without_mask)
    print("Pad mass with mask:", pad_mass_with_mask)

    assert pad_mass_with_mask == 0.0

    print("\nSteps 4 and 5 checks passed.")
    print("\nAll Step 2 checks passed.")


if __name__ == "__main__":
    main()