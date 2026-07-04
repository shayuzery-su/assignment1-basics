import collections
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse

import regex as re


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
        help="Path to the input training text file",
    )

    parser.add_argument(
        "vocab_size",
        type=int,
        help="Target vocabulary size",
    )

    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token to exclude from BPE training. Can be used multiple times.",
    )

    args = parser.parse_args()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
    )

    print("Vocabulary size:", len(vocab))
    print("Number of merges:", len(merges))
    print("First 10 merges:")
    for merge in merges[:10]:
        print(merge)