"""Data loading for tokenized tfrecord shards produced by data/*/prepare.py.

We concatenate all 'ids' (uint16) bytes from the tfrecord shards into a single
.bin file per split, then memmap it for random-access batch sampling (nanoGPT-style).
"""
import glob
import os
import numpy as np


def _iter_tfrecord_ids(tfrecord_paths):
    import tensorflow as tf

    for path in tfrecord_paths:
        for raw in tf.data.TFRecordDataset([path]):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            ids_bytes = ex.features.feature['ids'].bytes_list.value[0]
            yield np.frombuffer(ids_bytes, dtype=np.uint16)


def tfrecord_to_bin(tfrecord_glob: str, out_path: str):
    paths = sorted(glob.glob(tfrecord_glob))
    if not paths:
        raise FileNotFoundError(f"no tfrecord files match {tfrecord_glob}")
    total = 0
    with open(out_path, 'wb') as f:
        for arr in _iter_tfrecord_ids(paths):
            f.write(arr.tobytes())
            total += arr.size
    print(f"wrote {total:,} tokens to {out_path}")


def ensure_bin(data_dir: str, split: str) -> str:
    """Return path to <data_dir>/<split>.bin, building it from tfrecords if missing."""
    bin_path = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(bin_path):
        glob_pat = os.path.join(data_dir, f'{split}_*.tfrecord')
        tfrecord_to_bin(glob_pat, bin_path)
    return bin_path


class DataLoader:
    """Random-batch sampler over a uint16 memmap. Yields int32 batches of shape
    (batch_size, block_size + 1) — the LLM splits this into (x, y) internally."""

    def __init__(self, bin_path: str, batch_size: int, block_size: int, seed: int = 0):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        if len(self.data) < block_size + 2:
            raise ValueError(f"{bin_path} has only {len(self.data)} tokens, need > {block_size + 1}")
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        n = len(self.data) - self.block_size - 1
        ix = self.rng.integers(0, n, size=self.batch_size)
        batch = np.stack([
            np.asarray(self.data[i:i + self.block_size + 1], dtype=np.int32)
            for i in ix
        ])
        return batch
