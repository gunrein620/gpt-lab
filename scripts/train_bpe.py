from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CORPUS_PATH = DATA_DIR / "nsmc_lm_train.txt"
DEFAULT_VOCAB_PATH = DATA_DIR / "nsmc_bpe_vocab_4096.json"

sys.path.insert(0, str(SRC_DIR))

from bpe import BPETokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and sanity-check the byte-level BPE tokenizer."
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Text corpus path for BPE training.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Output JSON path for the trained BPE vocabulary.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Target tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=409_600,
        help="Maximum number of corpus characters to use for BPE training.",
    )
    parser.add_argument(
        "--sample",
        default="\uc774 \uc601\ud654\ub294 \uc88b\uc558\ub2e4. byte-level BPE test!",
        help="Text used for encode/decode round-trip check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = args.corpus_path
    vocab_path = args.vocab_path

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Create data files first, for example: python download_data.py"
        )

    corpus = corpus_path.read_text(encoding="utf-8")[: args.corpus_size]
    if len(corpus) < 2:
        raise ValueError("Corpus must contain at least 2 characters.")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    start = time.perf_counter()
    tokenizer.train(corpus)
    elapsed = time.perf_counter() - start

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(vocab_path)

    encoded = tokenizer.encode(args.sample)
    decoded = tokenizer.decode(encoded)
    round_trip_ok = decoded == args.sample

    print(f"vocab_size target: {args.vocab_size:,}")
    print(f"actual vocab size: {len(tokenizer.id_to_token):,}")
    print(f"merge count: {len(tokenizer.merges):,}")
    print(f"corpus chars used: {len(corpus):,}")
    print(f"train time sec: {elapsed:.2f}")
    print(f"vocab saved: {vocab_path}")
    print(f"sample token count: {len(encoded):,}")
    print(f"round trip ok: {round_trip_ok}")

    if not round_trip_ok:
        print(f"decoded sample: {decoded}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
