"""
Single-head self-attention demo (NumPy): tokenize, embed, add positional encoding,
compute attention, find the most "received" attention word, and plot the matrix.
"""

from __future__ import annotations

import argparse
import re
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np


# ---------------------------------------------------------------------------
# 1. Tokenization
# ---------------------------------------------------------------------------
def tokenize(sentence: str) -> List[str]:
    """Split sentence into word tokens (alphanumeric + apostrophe inside words)."""
    sentence = sentence.strip().lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", sentence)
    return tokens if tokens else sentence.split()


# ---------------------------------------------------------------------------
# 2. Word embeddings (learnable-style lookup; fixed random for demo)
# ---------------------------------------------------------------------------
def build_vocab(tokens: List[str]) -> dict:
    unique = sorted(set(tokens))
    return {w: i for i, w in enumerate(unique)}


def word_embeddings(
    tokens: List[str], vocab: dict, d_model: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sequence matrix (seq_len, d_model), full embedding table)."""
    vocab_size = len(vocab)
    table = rng.standard_normal((vocab_size, d_model)) * 0.5
    indices = [vocab[t] for t in tokens]
    return table[indices].copy(), table


# ---------------------------------------------------------------------------
# 3. Positional encoding (sinusoidal, Vaswani et al.)
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Shape (seq_len, d_model)."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len, dtype=np.float64)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


# ---------------------------------------------------------------------------
# 4. Single self-attention
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def single_head_self_attention(
    x: np.ndarray, rng: np.random.Generator, d_model: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    x: (seq_len, d_model)
    Returns: Q, K, V, scores (QK^T/sqrt(d_k)), attention_weights, output
    """
    seq_len, d = x.shape
    assert d == d_model

    # One projection matrix shared style: W_q, W_k, W_v
    scale = 1.0 / np.sqrt(d_model)
    w_q = rng.standard_normal((d_model, d_model)) * scale
    w_k = rng.standard_normal((d_model, d_model)) * scale
    w_v = rng.standard_normal((d_model, d_model)) * scale

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = d_model
    scores = (q @ k.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    out = attention_weights @ v
    return q, k, v, scores, attention_weights, out


# ---------------------------------------------------------------------------
# 6. Impact: total attention *received* per word (column sums)
# ---------------------------------------------------------------------------
def attention_received(attention_weights: np.ndarray) -> np.ndarray:
    """Sum over query positions: how much all words attend to each key position."""
    return attention_weights.sum(axis=0)


def most_influential_word(tokens: List[str], attention_weights: np.ndarray) -> Tuple[str, int, np.ndarray]:
    received = attention_received(attention_weights)
    j = int(np.argmax(received))
    return tokens[j], j, received


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
def plot_attention_matrix(tokens: List[str], attention_weights: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.6), max(5, len(tokens) * 0.55)))
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attends from)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def print_attention_matrix_console(tokens: List[str], attention_weights: np.ndarray, title: str) -> None:
    """NumPy-only relationship map: scaled values 0-9 in a text grid."""
    print(f"\n{title}\n")
    a = attention_weights
    lo, hi = float(a.min()), float(a.max())
    span = hi - lo if hi > lo else 1.0
    digits = ((a - lo) / span * 9).round().astype(int)
    col_w = max(5, max(len(t) for t in tokens) + 1)
    header = "".join(f"{t:>{col_w}}" for t in tokens)
    print(" " * 8 + header)
    for i, row_label in enumerate(tokens):
        row = "".join(f"{digits[i, j]:>{col_w}d}" for j in range(len(tokens)))
        print(f"{row_label:8s}{row}")
    print("\nCells are 0 (low attention) ... 9 (high) within this matrix; rows = query, cols = key.")


def print_step(title: str, body: str = "") -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    if body:
        print(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy single-head self-attention demo.")
    parser.add_argument(
        "sentence",
        nargs="?",
        default="The quick brown fox jumps over the lazy dog.",
        help="Input sentence.",
    )
    parser.add_argument("--d-model", type=int, default=16, help="Embedding / model dimension.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib window.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    d_model = args.d_model

    # 1
    print_step("1. Tokenization", f"Raw sentence:\n  {args.sentence!r}")
    tokens = tokenize(args.sentence)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # 2
    print_step("2. Word embedding", f"Dimension d_model = {d_model}")
    vocab = build_vocab(tokens)
    print(f"Vocabulary ({len(vocab)}): {list(vocab.keys())}")
    emb, _emb_table = word_embeddings(tokens, vocab, d_model, rng)
    print(f"Embedding matrix X shape: {emb.shape}  (seq_len x d_model)")
    print(f"First token vector (first 8 dims): {np.round(emb[0, :8], 4)}")

    # 3
    print_step("3. Positional encoding")
    pe = positional_encoding(len(tokens), d_model)
    x = emb + pe
    print(f"PE shape: {pe.shape}; X' = embeddings + PE, shape: {x.shape}")
    print(f"PE row 0 (first 8 dims): {np.round(pe[0, :8], 4)}")

    # 4
    print_step(
        "4. Single self-attention",
        "Compute Q = X'W_q, K = X'W_k, V = X'W_v;\n"
        "scores = Q K^T / sqrt(d_k); weights = softmax(scores, row-wise); output = weights @ V.",
    )
    q, k, v, scores, attn, out = single_head_self_attention(x, rng, d_model)
    print(f"Q, K, V shapes: {q.shape}, {k.shape}, {v.shape}")
    print(f"Score matrix S = QK^T / sqrt(d_k) shape: {scores.shape}")
    print(f"Attention weights A (after softmax on each row) shape: {attn.shape}")
    print(f"Self-attention output shape: {out.shape}")
    print("\nAttention matrix A (rows = query word, cols = key word):")
    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print(attn)

    # 5
    print_step(
        "5. Relationship map",
        "A[i, j] = how much token i (query) attends to token j (key).\n"
        "Rows sum to 1 (each word distributes its attention across all keys).",
    )

    # 6
    received = attention_received(attn)
    word, idx, _ = most_influential_word(tokens, attn)
    print_step(
        "6. Impact analysis",
        "Total attention *received* by each word j: sum_i A[i, j]\n"
        f"(which word other positions attend to the most, in aggregate).\n\n"
        f"Per-token received totals:\n  {dict(zip(tokens, np.round(received, 4)))}\n\n"
        f"Most influential word (highest received attention): '{word}' (index {idx}).",
    )

    # 7
    want_gui = not args.no_plot and _HAS_MPL
    print_step(
        "7. Visualization",
        "Heatmap (GUI) if matplotlib is installed; otherwise a 0-9 console grid."
        if not args.no_plot
        else "Console relationship map (--no-plot).",
    )
    if want_gui:
        plot_attention_matrix(
            tokens,
            attn,
            title="Self-attention weights: query (row) -> key (column)",
        )
    else:
        if not args.no_plot and not _HAS_MPL:
            print("Note: matplotlib not installed — using text grid. pip install matplotlib for a figure.\n")
        print_attention_matrix_console(
            tokens,
            attn,
            title="Self-attention weights: query (row) -> key (column)",
        )


if __name__ == "__main__":
    main()
