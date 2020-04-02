from torch import nn

from constants.constants import *
from sesame_street.models.Bert import Bert


class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.bert = Bert(EMBED_SIZE, WORD_VOCAB_SIZE, TYPE_VOCAB_SIZE, POS_VOCAB_SIZE, NUM_LAYERS)
        self.out = nn.Linear(EMBED_SIZE, NUM_CLASSES)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT)


    def forward(self, input_ids, token_ids, mask_ids = None):
        bert_outputs = self.bert(input_ids, token_ids, mask_ids)
        bert_outputs = self.dropout(bert_outputs)
        logits = self.out(bert_outputs)
        return logits
