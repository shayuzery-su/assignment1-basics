import argparse
import collections
import json
import multiprocessing as mp
import os
import pickle
import time
import tracemalloc
from pathlib import Path

import regex as re


# GPT-2-style pre-tokenization regex.
# Python's built-in re module does not support \p{L} and \p{N},
# so this script uses the third-party regex package.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


TokenId = int
Word = tuple[TokenId, ...]
Pair = tuple[TokenId, TokenId]


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text on special tokens.

    The special tokens themselves are removed from the returned parts.
    This prevents BPE merges from crossing special-token boundaries.
    """
    if not special_tokens:
        return [text]

    escaped_tokens = [
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    ]

    delimiter_pattern = "|".join(escaped_tokens)
    return re.split(delimiter_pattern, text)


def find_chunk_boundaries(
    input_path: str | Path,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Split a file into byte ranges whose boundaries occur at split_special_token.

    This is useful for parallel pre-tokenization. Each worker receives a chunk
    that ends at a document boundary, so pre-tokenization cannot accidentally
    merge text across documents.
    """
    if not split_special_token:
        raise ValueError("split_special_token must not be empty")

    input_path = Path(input_path)

    with input_path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if desired_num_chunks <= 1 or file_size == 0:
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks

        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        overlap_size = len(split_special_token) - 1

        for boundary_index in range(1, len(boundaries) - 1):
            file.seek(boundaries[boundary_index])
            previous_tail = b""

            while True:
                current_position = file.tell()
                mini_chunk = file.read(mini_chunk_size)

                if mini_chunk == b"":
                    boundaries[boundary_index] = file_size
                    break

                search_chunk = previous_tail + mini_chunk
                found_at = search_chunk.find(split_special_token)

                if found_at != -1:
                    boundaries[boundary_index] = (
                        current_position - len(previous_tail) + found_at
                    )
                    break

                previous_tail = (
                    search_chunk[-overlap_size:]
                    if overlap_size > 0
                    else b""
                )

    return sorted(set(boundaries))


def pretokenize_text(text: str) -> collections.Counter[Word]:
    """
    Pre-tokenize text using PAT.

    Each regex pre-token is encoded to UTF-8 bytes.
    Internally, each byte is represented by its integer value, 0..255.

    Example:
        "hello" -> (104, 101, 108, 108, 111)
    """
    counts: collections.Counter[Word] = collections.Counter()

    for match in re.finditer(PAT, text):
        token_bytes = match.group(0).encode("utf-8")

        if token_bytes:
            counts[tuple(token_bytes)] += 1

    return counts


def pretokenize_file_chunk(
    args: tuple[str, int, int, tuple[str, ...]],
) -> collections.Counter[Word]:
    """
    Worker function for multiprocessing.

    Reads one byte range from the input file, removes/splits on special tokens,
    and returns pre-token counts for that chunk.
    """
    input_path, start, end, special_tokens_tuple = args
    special_tokens = list(special_tokens_tuple)

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="strict")

    total_counts: collections.Counter[Word] = collections.Counter()

    for part in split_on_special_tokens(text, special_tokens):
        if part:
            total_counts.update(pretokenize_text(part))

    return total_counts


def build_pretoken_counts_single_process(
    input_path: str | Path,
    special_tokens: list[str],
) -> collections.Counter[Word]:
    """
    Build pre-token counts without multiprocessing.
    """
    input_path = Path(input_path)
    text = input_path.read_text(encoding="utf-8")

    total_counts: collections.Counter[Word] = collections.Counter()

    for part in split_on_special_tokens(text, special_tokens):
        if part:
            total_counts.update(pretokenize_text(part))

    return total_counts


def build_pretoken_counts(
    input_path: str | Path,
    special_tokens: list[str],
    num_processes: int,
) -> collections.Counter[Word]:
    """
    Build pre-token counts.

    If num_processes > 1 and special tokens are available, this parallelizes
    pre-tokenization by splitting the file at the first special token.
    Otherwise, it falls back to a single-process implementation.
    """
    input_path = Path(input_path)

    if num_processes <= 1 or not special_tokens:
        return build_pretoken_counts_single_process(
            input_path=input_path,
            special_tokens=special_tokens,
        )

    split_token = special_tokens[0].encode("utf-8")

    boundaries = find_chunk_boundaries(
        input_path=input_path,
        desired_num_chunks=num_processes,
        split_special_token=split_token,
    )

    tasks = [
        (str(input_path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    if not tasks:
        return collections.Counter()

    total_counts: collections.Counter[Word] = collections.Counter()

    with mp.Pool(processes=min(num_processes, len(tasks))) as pool:
        for partial_counts in pool.map(pretokenize_file_chunk, tasks):
            total_counts.update(partial_counts)

    return total_counts


def contains_pair(word: Word, pair: Pair) -> bool:
    """
    Return True if word contains pair as adjacent tokens.
    """
    left, right = pair

    for i in range(len(word) - 1):
        if word[i] == left and word[i + 1] == right:
            return True

    return False


def merge_word(word: Word, pair: Pair, new_token_id: int) -> Word:
    """
    Replace every non-overlapping occurrence of pair with new_token_id.
    """
    left, right = pair
    result: list[int] = []

    i = 0
    while i < len(word):
        if (
            i < len(word) - 1
            and word[i] == left
            and word[i + 1] == right
        ):
            result.append(new_token_id)
            i += 2
        else:
            result.append(word[i])
            i += 1

    return tuple(result)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
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

        num_processes:
            Number of processes to use for pre-tokenization.
            The BPE merge loop itself remains single-process.

    Returns:
        vocab:
            dict[int, bytes], mapping token ID to token bytes.

        merges:
            list[tuple[bytes, bytes]], ordered list of BPE merges.
    """
    input_path = Path(input_path)

    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(256)
    }

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than initial vocabulary size={len(vocab)}"
        )

    merges: list[tuple[bytes, bytes]] = []

    word_counts = build_pretoken_counts(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )

    words: list[Word] = list(word_counts.keys())
    counts: list[int] = [word_counts[word] for word in words]

    pair_counts: collections.Counter[Pair] = collections.Counter()
    pair_to_word_ids: dict[Pair, set[int]] = collections.defaultdict(set)

    for word_id, word in enumerate(words):
        count = counts[word_id]

        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(word_id)

    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        affected_word_ids = [
            word_id
            for word_id in pair_to_word_ids.get(best_pair, set())
            if contains_pair(words[word_id], best_pair)
        ]

        if not affected_word_ids:
            del pair_counts[best_pair]
            continue

        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]

        merges.append((left_bytes, right_bytes))

        new_token_id = len(vocab)
        vocab[new_token_id] = left_bytes + right_bytes

        for word_id in affected_word_ids:
            old_word = words[word_id]
            count = counts[word_id]

            for old_pair in zip(old_word, old_word[1:]):
                pair_counts[old_pair] -= count

                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

                pair_to_word_ids[old_pair].discard(word_id)

            new_word = merge_word(old_word, best_pair, new_token_id)
            words[word_id] = new_word

            for new_pair in zip(new_word, new_word[1:]):
                pair_counts[new_pair] += count
                pair_to_word_ids[new_pair].add(word_id)

    return vocab, merges


def serialize_bpe_result(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | Path,
    elapsed_seconds: float,
    peak_memory_mb: float,
) -> None:
    """
    Save trained BPE artifacts to disk.

    Saves:
    - vocab.pkl: exact Python dict[int, bytes]
    - merges.pkl: exact Python list[tuple[bytes, bytes]]
    - vocab.json: human-readable vocabulary
    - merges.json: human-readable merges
    - summary.json: training summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)

    with (output_dir / "merges.pkl").open("wb") as file:
        pickle.dump(merges, file)

    vocab_json = {
        str(token_id): {
            "hex": token_bytes.hex(),
            "repr": repr(token_bytes),
            "utf8_preview": token_bytes.decode("utf-8", errors="replace"),
            "length_bytes": len(token_bytes),
        }
        for token_id, token_bytes in vocab.items()
    }

    with (output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab_json, file, ensure_ascii=False, indent=2)

    merges_json = [
        {
            "rank": rank,
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_repr": repr(left),
            "right_repr": repr(right),
            "merged_hex": (left + right).hex(),
            "merged_repr": repr(left + right),
            "merged_utf8_preview": (left + right).decode("utf-8", errors="replace"),
            "merged_length_bytes": len(left + right),
        }
        for rank, (left, right) in enumerate(merges)
    ]

    with (output_dir / "merges.json").open("w", encoding="utf-8") as file:
        json.dump(merges_json, file, ensure_ascii=False, indent=2)

    longest_token_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    summary = {
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb_tracemalloc": peak_memory_mb,
        "longest_token_id": longest_token_id,
        "longest_token_length_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "longest_token_utf8_preview": longest_token.decode("utf-8", errors="replace"),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Saved artifacts to: {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Peak traced Python memory: {peak_memory_mb:.2f} MB")
    print(f"Longest token ID: {longest_token_id}")
    print(f"Longest token: {longest_token.decode('utf-8', errors='replace')!r}")
    print(f"Longest token length: {len(longest_token)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer"
    )

    parser.add_argument(
        "input_path",
        help="Path to the training text file",
    )

    parser.add_argument(
        "vocab_size",
        type=int,
        help="Maximum final vocabulary size",
    )

    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token to add to the vocabulary. Can be used multiple times.",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of processes to use for pre-tokenization",
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
        num_processes=args.num_processes,
    )

    elapsed_seconds = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    serialize_bpe_result(
        vocab=vocab,
        merges=merges,
        output_dir=args.output_dir,
        elapsed_seconds=elapsed_seconds,
        peak_memory_mb=peak_memory / 1024 / 1024,
    )


if __name__ == "__main__":
    main()