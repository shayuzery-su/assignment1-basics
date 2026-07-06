import collections
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse

import regex as re

import argparse
import json
import pickle
import time
import tracemalloc
from pathlib import Path


# GPT-2-style pre-tokenization regex.
# The normal Python "re" module does not support \p{L} and \p{N},
# so this uses the third-party "regex" package.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text on special tokens.

    The special tokens themselves are removed from the returned parts.
    This prevents BPE merges from crossing special-token boundaries.
    """
    if not special_tokens:
        return [text]

    # Longer tokens first avoids partial matches when one special token
    # is a prefix of another.
    escaped_tokens = [
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    ]

    delimiter_pattern = "|".join(escaped_tokens)

    return re.split(delimiter_pattern, text)


def pretokenize(text: str) -> collections.Counter[Pretoken]:
    """
    Convert text into counted pre-tokens.

    Each regex pre-token is encoded as UTF-8 bytes, then represented as a tuple
    of one-byte bytes objects.

    Example:
        "hello" -> (b"h", b"e", b"l", b"l", b"o")
    """
    counts: collections.Counter[Pretoken] = collections.Counter()

    for match in re.finditer(PAT, text):
        token_text = match.group(0)
        token_bytes = token_text.encode("utf-8")
        byte_tokens = tuple(bytes([b]) for b in token_bytes)

        if byte_tokens:
            counts[byte_tokens] += 1

    return counts


def get_pair_counts(
    pretoken_counts: dict[Pretoken, int],
) -> collections.Counter[Pair]:
    """
    Count adjacent token pairs across all pre-tokens.
    """
    pair_counts: collections.Counter[Pair] = collections.Counter()

    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count

    return pair_counts


def merge_pair_in_pretoken(pretoken: Pretoken, pair: Pair) -> Pretoken:
    """
    Replace every non-overlapping occurrence of pair with its merged token.

    Example:
        pretoken = (b"a", b"b", b"a", b"b")
        pair = (b"a", b"b")

        result = (b"ab", b"ab")
    """
    merged_token = pair[0] + pair[1]
    result: list[Token] = []

    i = 0
    while i < len(pretoken):
        if (
            i < len(pretoken) - 1
            and pretoken[i] == pair[0]
            and pretoken[i + 1] == pair[1]
        ):
            result.append(merged_token)
            i += 2
        else:
            result.append(pretoken[i])
            i += 1

    return tuple(result)


def merge_pair(
    pretoken_counts: dict[Pretoken, int],
    pair: Pair,
) -> dict[Pretoken, int]:
    """
    Apply one BPE merge to all pre-tokens.
    """
    new_counts: collections.Counter[Pretoken] = collections.Counter()

    for pretoken, count in pretoken_counts.items():
        new_pretoken = merge_pair_in_pretoken(pretoken, pair)
        new_counts[new_pretoken] += count

    return dict(new_counts)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path:
            Path to the input text file.

        vocab_size:
            Maximum final vocabulary size, including:
            - the initial 256 byte tokens
            - special tokens
            - tokens created by BPE merges

        special_tokens:
            Special tokens to add to the vocabulary.
            During training, they are treated as hard split boundaries and
            are not included in merge statistics.

    Returns:
        vocab:
            dict[int, bytes], mapping token ID to token bytes.

        merges:
            list[tuple[bytes, bytes]], ordered list of BPE merges.
    """
    input_path = Path(input_path)

    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            "vocab_size must be at least 256 + len(special_tokens)"
        )

    # Initial byte vocabulary: token IDs 0..255.
    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(256)
    }

    # Add special tokens after the 256 byte tokens.
    next_token_id = 256
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    # Read text.
    text = input_path.read_text(encoding="utf-8")

    # Split on special tokens so merges cannot cross them.
    text_parts = split_on_special_tokens(text, special_tokens)

    # Pre-tokenize each part separately.
    pretoken_counts: collections.Counter[Pretoken] = collections.Counter()

    for part in text_parts:
        if part:
            pretoken_counts.update(pretokenize(part))

    # Train merges until vocabulary reaches requested size.
    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(pretoken_counts)

        if not pair_counts:
            break

        # Pick the most frequent pair.
        # If there is a tie, this chooses the lexicographically largest pair.
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        pretoken_counts = collections.Counter(
            merge_pair(pretoken_counts, best_pair)
        )

    return vocab, merges

def serialize_bpe_result(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | Path,
    elapsed_seconds: float | None = None,
    peak_memory_mb: float | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exact Python serialization
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Human-readable vocabulary
    vocab_json = {
        str(token_id): {
            "hex": token_bytes.hex(),
            "repr": repr(token_bytes),
            "utf8_preview": token_bytes.decode("utf-8", errors="replace"),
            "length_bytes": len(token_bytes),
        }
        for token_id, token_bytes in vocab.items()
    }

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Human-readable merges
    merges_json = [
        {
            "rank": i,
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_repr": repr(left),
            "right_repr": repr(right),
            "merged_repr": repr(left + right),
            "merged_utf8_preview": (left + right).decode("utf-8", errors="replace"),
            "merged_length_bytes": len(left + right),
        }
        for i, (left, right) in enumerate(merges)
    ]

    with open(output_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)

    # Longest token
    longest_token_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    summary = {
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": longest_token_id,
        "longest_token_length_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "longest_token_utf8_preview": longest_token.decode("utf-8", errors="replace"),
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb_tracemalloc": peak_memory_mb,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved BPE artifacts to: {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token ID: {longest_token_id}")
    print(f"Longest token length in bytes: {len(longest_token)}")
    print(f"Longest token preview: {longest_token.decode('utf-8', errors='replace')!r}")

# Optional adapter function, useful if your assignment expects this name.
def run_train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(input_path, vocab_size, special_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")

    parser.add_argument(
        "input_path",
        help="Path to the training text file",
    )

    parser.add_argument(
        "vocab_size",
        type=int,
        help="Maximum vocabulary size",
    )

    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token to add to the vocabulary. Can be used multiple times.",
    )

    parser.add_argument(
        "--output-dir",
        default="bpe_artifacts",
        help="Directory where vocab and merges will be saved",
    )

    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
    )

    elapsed_seconds = time.perf_counter() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_memory / 1024 / 1024

    serialize_bpe_result(
        vocab=vocab,
        merges=merges,
        output_dir=args.output_dir,
        elapsed_seconds=elapsed_seconds,
        peak_memory_mb=peak_memory_mb,
    )

    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Peak traced Python memory: {peak_memory_mb:.2f} MB")