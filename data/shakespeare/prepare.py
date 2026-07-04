"""
Copied from NanoGPT: 
    https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/data/shakespeare/prepare.py 
Adapted to tfds from 
Tiny shakespeare, of the good old char-rnn fame :)

After running `prepare.py`:

    train.bin has 301,966 tokens
    val.bin has 36,059 tokens

"""
from multiprocessing import cpu_count
from tqdm import tqdm
import os
import requests
import tiktoken
import numpy as np
import tensorflow as tf

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
tokenized = {'train_01':[train_ids], 'val_01':[val_ids]}
print(f"train has {len(train_ids):,} tokens") # train.bin has 301,966 tokens
print(f"val has {len(val_ids):,} tokens") # val.bin has 36,059 tokens

# # export to bin files
# train_ids = np.array(train_ids, dtype=np.uint16)
# val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# export to bin files with tensorflow
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for split, dset in tokenized.items():
    filename = f'{split}.tfrecord'

    print(f"writing {filename}...")
    with tf.io.TFRecordWriter(filename) as writer:
        for example in tqdm(dset):
            feature = np.asarray(example, dtype=np.uint16).tobytes()
            example_proto = tf.train.Example(
                features=tf.train.Features(feature={
                    'ids': _bytes_feature(feature)
                    }))
            writer.write(example_proto.SerializeToString())