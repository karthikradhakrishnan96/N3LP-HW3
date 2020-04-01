import json

from torchtext.data import Example, Field, Dataset


class RumorDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.fields = [
            ('input_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('token_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('mask_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('labels', Field(sequential=False, use_vocab=False, batch_first=True, is_target=True))
        ]
        with open(path) as f:
            data = json.load(f)
        self.prep_data(data)
        super().__init__(self.examples, self.fields)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def prep_data(self, data):
        examples = []
        for example in data['Examples']:
            target = example["spacy_processed_text"]
            parent = example["spacy_processed_text_prev"]
            source = example["spacy_processed_text_src"]

            target = self.tokenizer.tokens_to_ids(self.tokenizer.tokenize(target))
            parent = self.tokenizer.tokens_to_ids(self.tokenizer.tokenize(parent))
            source = self.tokenizer.tokens_to_ids(self.tokenizer.tokenize(source))

            context = source + parent
            cls_id = [self.tokenizer.vocab["[CLS]"]]
            sep_id = [self.tokenizer.vocab["[SEP]"]]
            input_ids = cls_id + context + sep_id + target + sep_id

            max_length = 512-3

            if len(input_ids) > max_length:
                context = context[:max_length // 2]
                input_ids = cls_id + context + sep_id + target + sep_id
                if len(input_ids) > max_length:
                    target = target[:max_length // 2]
                    input_ids = cls_id + context + sep_id + target + sep_id

            token_ids = [0] * (len(context) + 2) + [1] * (len(target) + 1)
            label = example['stance_label']
            mask_ids = [1] * len(input_ids)
            example = Example.fromlist([input_ids, token_ids, mask_ids, label], self.fields)
            examples.append(example)
        self.examples = examples
