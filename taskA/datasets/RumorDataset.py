import json

from torchtext.data import Example, Field, Dataset


class RumorDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.fields = [
            ('input_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('token_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('mask_ids', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('input_ids2', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('token_ids2', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('mask_ids2', Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
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


    def _make_ids(self, context, target):
        context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
        target = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(target))
        cls_id = [self.tokenizer.vocab["[CLS]"]]
        sep_id = [self.tokenizer.vocab["[SEP]"]]
        input_ids = cls_id + context + sep_id + target + sep_id

        max_length = 512 - 3

        if len(input_ids) > max_length:
            context = context[:max_length // 2]
            input_ids = cls_id + context + sep_id + target + sep_id
            if len(input_ids) > max_length:
                target = target[:max_length // 2]
                input_ids = cls_id + context + sep_id + target + sep_id

        token_ids = [0] * (len(context) + 2) + [1] * (len(target) + 1)
        mask_ids = [1] * len(input_ids)
        return input_ids, token_ids, mask_ids

    def prep_data(self, data):
        examples = []
        for example in data['Examples']:
            target = example["spacy_processed_text"]
            parent = example["spacy_processed_text_prev"]
            source = example["spacy_processed_text_src"]

            input_ids, token_ids, mask_ids = self._make_ids(parent, target)
            input_ids2, token_ids2, mask_ids2 = self._make_ids(source, parent)

            label = example['stance_label']
            example = Example.fromlist([input_ids, token_ids, mask_ids, input_ids2, token_ids2, mask_ids2, label], self.fields)
            examples.append(example)
        self.examples = examples
