import json
import re

import chex
from data.rapanui.corpus.make_dataset import normalize_rapanui


def remove_number_dot_prefix(text):
    """
    Removes a leading number and dot (e.g., '22. ') from the start of the string.
    """
    return re.sub(r'^\d+\.\s*', '', text)


def normalize_rapanui_full(text):
    text = normalize_rapanui(text)
    text = remove_number_dot_prefix(text)
    return text


class BaseTokenizer(object):
    def __init__(self, tablet_path: str, special_tokens: list[str] = None):
        self.tablet = json.load(open(tablet_path))
        self.vocab = set()
        self.special_tokens = special_tokens if special_tokens else []
        self.base_split_pattern = None  # To be set by child classes
    
    def _build_vocab(self):
        raise NotImplementedError
    
    def add_special_tokens(self, tokens: list[str]):
        """Add special tokens to the vocabulary and update mappings."""
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)
        
        # Add special tokens to vocab
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab.append(token)
        
        # Update mappings
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Update split pattern to recognize special tokens
        self._update_split_pattern()
    
    def _finalize_vocab(self):
        """Convert vocab to list and add special tokens, then create mappings."""
        if isinstance(self.vocab, set):
            self.vocab = list(self.vocab)
        
        # Add special tokens at the beginning of vocab
        for token in reversed(self.special_tokens):
            if token not in self.vocab:
                self.vocab.insert(0, token)
        
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Update split pattern to recognize special tokens
        self._update_split_pattern()
    
    def _update_split_pattern(self):
        """Update the split pattern to include special tokens."""
        if self.special_tokens and self.base_split_pattern:
            # Escape special tokens for regex and join with |
            escaped_special = [re.escape(token) for token in self.special_tokens]
            special_pattern = '|'.join(escaped_special)
            # Special tokens should be matched first
            self.split_pattern = f'{special_pattern}|{self.base_split_pattern}'
        else:
            self.split_pattern = self.base_split_pattern

    def decode(self, tokens: list[int]):
        return ''.join([self.id_to_token[token] for token in tokens])
    
    def encode_batch(self, text: list[str]):
        return [self.encode(t) for t in text]
    
    def encode(self, text: str):
        tokens = re.findall(self.split_pattern, text)
        tokens = [t for t in tokens if t]
        return [self.token_to_id[token] for token in tokens]
    
    def decode_batch(self, tokens: list[list[int]]):
        return [self.decode(t) for t in tokens]

    def save(self, path: str):
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_tokens': self.special_tokens
        }   
        json.dump(data, open(path, "w"))

    def load(self, path: str):
        data = json.load(open(path, "r"))
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        # Ensure that id keys are ints (JSON serialization can turn them into strings)
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.special_tokens = data.get('special_tokens', [])


class RapanuiTokenizer(BaseTokenizer):
    def __init__(self, tablet_path: str, special_tokens: list[str] = None):
        super().__init__(tablet_path, special_tokens)
        self.base_split_pattern = r'\w+|[^\w\s]'
        self._build_vocab()

    def _build_vocab(self):
        for lines in self.tablet:
            for line in lines:
                if line != '. . .':
                    line = normalize_rapanui(line)
                    line = remove_number_dot_prefix(line)
                    # Extract words and punctuation separately
                    tokens = re.findall(self.base_split_pattern, line)
                    # Filter out empty strings
                    tokens = [t for t in tokens if t]
                    self.vocab.update(tokens)
        self._finalize_vocab()


class RoRTokenizer(BaseTokenizer):

    def __init__(self, tablet_path: str, special_tokens: list[str] = None):
        super().__init__(tablet_path, special_tokens)
        self.base_split_pattern = r'\w+|[^\w\s]'
        self._build_vocab()

    def _build_vocab(self):
        for name, tablet in self.tablet.items():
            for line, text in tablet.items():
                # Extract words and punctuation separately
                tokens = re.findall(self.base_split_pattern, text)
                # Filter out empty strings
                tokens = [t for t in tokens if t]
                self.vocab.update(tokens)
        self._finalize_vocab()


class CommonTokenizer(BaseTokenizer):
    def __init__(self, ror_path, rapanui_path, special_tokens: list[str] = None):
        # Initialize without calling super().__init__ since we don't have a single tablet_path
        self.special_tokens = special_tokens if special_tokens else []
        self.base_split_pattern = r'\w+|[^\w\s]'
        
        # Create individual tokenizers (they get their own special tokens)
        self.ror_tokenizer = RoRTokenizer(ror_path, special_tokens)
        self.rapanui_tokenizer = RapanuiTokenizer(rapanui_path, special_tokens)
        
        # Combine vocabularies
        self.vocab = set(self.ror_tokenizer.vocab)
        self.vocab.update(self.rapanui_tokenizer.vocab)
        self.vocab = list(self.vocab)
        
        # Ensure special tokens are at the beginning
        for token in reversed(self.special_tokens):
            if token in self.vocab:
                self.vocab.remove(token)
            self.vocab.insert(0, token)
        
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Update split pattern to recognize special tokens
        self._update_split_pattern()
