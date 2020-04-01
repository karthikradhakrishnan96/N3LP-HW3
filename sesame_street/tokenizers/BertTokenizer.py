from sesame_street.constants.file_paths import VOCAB_PATH
import os
import urllib.request
import unicodedata



PRETRAINED_VOCABS = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    }


def load(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            vocab[word.strip()] = idx
    return vocab


def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    return "".join(ch for ch in text if unicodedata.category(ch) != 'Mn')


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C')


class BertTokenizer:
    def __init__(self, vocab_path=None, do_lower_case=True):
        if not os.path.isfile(vocab_path):
            raise ValueError('{} is not a valid vocabulary file'.format(vocab_path))
        self.vocab = load(vocab_path)
        self.rev_vocab = {self.vocab[k]:k for k in self.vocab}
        self.word_piece_tokenizer = WordPieceTokenizer(self.vocab)
        self.word_level_tokenizer = WordLevelTokenizer(convert_to_lower_case=do_lower_case)

    def tokenize(self, text):
        tokens = []
        for token in self.word_level_tokenizer.tokenize(text):
            tokens.extend(self.word_piece_tokenizer.tokenize(token))
        return tokens

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.rev_vocab[id] for id in ids]

    @classmethod
    def from_pretrained(cls, model_name, do_lower_case=True):
        if model_name in PRETRAINED_VOCABS:
            vocab = PRETRAINED_VOCABS[model_name]
            urllib.request.urlretrieve(vocab, VOCAB_PATH)
            vocab = VOCAB_PATH
        else:
            vocab = os.path.join(model_name, VOCAB_PATH) if os.path.isdir(model_name) else model_name
        return cls(vocab, do_lower_case)


def split_on_punctuation(text):
    tokens = []
    text = list(text)
    split = True
    for i in text:
        if not i.isdigit() and not i.isalpha():
            tokens.append([i])
            split = True
        else:
            if split:
                tokens.append([])
                split = False
            tokens[-1].append(i)
    return [''.join(i) for i in tokens]


class WordLevelTokenizer():
    def __init__(self, convert_to_lower_case):
        self.convert_to_lower = convert_to_lower_case

    def tokenize(self, text):
        tokens = text.split()
        output_tokens = []
        for tok in tokens:
            tok = remove_control_characters(tok)
            if self.convert_to_lower:
                tok = strip_accents(tok.lower())
            output_tokens.extend(split_on_punctuation(tok))
        return output_tokens


class WordPieceTokenizer():
    def __init__(self, vocab, unk_token='[UNK]', max_word_len=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
        self.prefix = '##'

    def tokenize(self, text):
        tokens = []
        if len(text) > self.max_word_len:
            tokens.append(self.unk_token)
        else:
            characters = list(text)
            start = 0
            is_invalid = False
            while start < len(characters) and not is_invalid:
                end = len(characters)
                curr_token = None
                while start < end:
                    token = ''.join(characters[start:end])
                    """
                    Add ## if it's not the first token
                    """
                    if start != 0:
                        token = self.prefix + token
                    if token in self.vocab:
                        curr_token = token
                        break
                    end -= 1
                if not curr_token:
                    is_invalid = True
                else:
                    tokens.append(curr_token)
                start = end
            if is_invalid:
                tokens = [self.unk_token]
        return tokens