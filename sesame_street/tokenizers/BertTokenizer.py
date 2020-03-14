from constants.file_paths import VOCAB_PATH
import logging
import torch.nn as nn

from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer


class BertTokenizer:
    def __init__(self, vocab_path = None):
        if not vocab_path:
            vocab_path = VOCAB_PATH
        self.vocab = self.load(vocab_path)
        self.rev_vocab = {self.vocab[k]:k for k in self.vocab}

    def load(self, vocab_path):
        vocab = {}
        with open(vocab_path, 'r') as f:
            for idx, word in enumerate(f):
                vocab[word.strip()] = idx
        return vocab

    def tokenize(self, text):
        pass

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.rev_vocab[id] for id in ids]





if __name__ == "__main__":
    b = BertTokenizer()
    print("hi")