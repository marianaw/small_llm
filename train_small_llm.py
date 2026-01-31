#!/usr/bin/env python3
import argparse
import json
import os
from typing import Iterator, List, Tuple

import numpy as np
import jax.numpy as jnp

from small_llm.tokenizer import BaseTokenizer, CommonTokenizer, RoRTokenizer, normalize_rapanui_full
from small_llm.llm import LLM, ModelConfig

ROR_START_TOKEN = "<ROR_START>"
ROR_END_TOKEN = "<ROR_END>"
RAPANUI_START_TOKEN = "<RAPANUI_START>"
RAPANUI_END_TOKEN = "<RAPANUI_END>"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ror_corpus_with_markers(tablets: dict, start_tok: str, end_tok: str) -> str:
    # Sort tablets by key for deterministic order
    tablet_keys = sorted(tablets.keys())
    pieces: List[str] = []
    for tkey in tablet_keys:
        lines = tablets[tkey]
        # Sort lines to keep a stable order within a tablet
        line_keys = sorted(lines.keys())
        content = "-".join([lines[k] for k in line_keys])
        # Surround with separators so tokenizer sees markers as standalone tokens
        pieces.append(f"{start_tok}-{content}-{end_tok}")
    return "-".join(pieces)


def build_rapanui_corpus_with_markers(rapanui: list[list[str]], start_tok: str, end_tok: str) -> str:
    # Sort tablets by key for deterministic order
    pieces: List[str] = []
    for lines in rapanui:
        normalized_lines = []
        for line in lines:
            normalized_lines.append(normalize_rapanui_full(line))
        content = "\n".join(normalized_lines)
        pieces.append(f"{start_tok} {content} {end_tok}")
    return "-".join(pieces)


def encode_corpus(tok: BaseTokenizer, corpus: str) -> np.ndarray:
    ids = tok.encode(corpus)
    return np.array(ids, dtype=np.int32)


def make_stream(data: np.ndarray, seq_len: int, batch_size: int, seed: int) -> Iterator[jnp.ndarray]:
    rng = np.random.RandomState(seed)
    # Batch consists of sequences of length (seq_len+1) for next-token targets inside the model
    max_start = len(data) - (seq_len + 1)
    idxs = np.arange(max_start)

    while True:
        rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i + batch_size]
            if len(batch_idx) < batch_size:
                continue
            batch = np.stack([data[j:j + seq_len + 1] for j in batch_idx], axis=0)
            yield jnp.array(batch, dtype=jnp.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_llm.json", type=str)
    parser.add_argument("--tablets", default=os.path.join("data", "rongorongo", "codes", "tablets.json"), type=str)
    parser.add_argument("--rapanui", default=os.path.join("data", "rapanui", "corpus", "corpus_monolingual.json"), type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--val_frac", default=0.05, type=float)
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # Define special tokens
    special_tokens = [ROR_START_TOKEN, ROR_END_TOKEN, RAPANUI_START_TOKEN, RAPANUI_END_TOKEN]
    
    # Build tokenizer from dataset file with special tokens
    tokenizer = CommonTokenizer(args.tablets, args.rapanui, special_tokens=special_tokens)
    
    # Build corpus with explicit markers
    tablets = load_json(args.tablets)
    rapanui = load_json(args.rapanui)
    ror_corpus = build_ror_corpus_with_markers(tablets, start_tok=ROR_START_TOKEN, end_tok=ROR_END_TOKEN)
    rapanui_corpus = build_rapanui_corpus_with_markers(rapanui, start_tok=RAPANUI_START_TOKEN, end_tok=RAPANUI_END_TOKEN)
    corpus = '-'.join([ror_corpus, rapanui_corpus])

    # Encode corpus (special tokens are already in vocab and will be properly recognized)
    ids = encode_corpus(tokenizer, corpus)

    # Load config and model
    config = ModelConfig.from_json(args.config)
    config.vocab_size = tokenizer.vocab_size
    llm = LLM(config)
    if args.checkpoint:
        llm.load(args.checkpoint)
    seq_len = min(llm.config.max_len, 1024)

    # Train/val split
    n = len(ids)
    val_n = int(n * args.val_frac)
    train_ids = ids[:-val_n] if val_n > 0 else ids
    val_ids = ids[-val_n:] if val_n > 0 else ids[: min(1000, len(ids))]

    train_stream = make_stream(train_ids, seq_len=seq_len, batch_size=args.batch_size, seed=llm.config.seed)
    val_stream = make_stream(val_ids, seq_len=seq_len, batch_size=args.batch_size, seed=llm.config.seed + 1)

    # Simple training loop
    llm.train(train_stream, val_stream, args.steps)

    # Save
    llm.save("model_combined.pkl")
    tokenizer.save("tokenizer_combined.json")

    # Example generation from the last few tokens of train set
    prompt = jnp.array(train_ids[: seq_len][None, :], dtype=jnp.int32)
    gen = llm.generate(prompt)
    print("Generated token ids:", np.array(gen)[0, -llm.config.sample_config.max_new_tokens:])


if __name__ == "__main__":
    main()
