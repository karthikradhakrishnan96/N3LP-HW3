from torch import nn

from constants.model_consts import *
from sesame_street.models.Bert import Bert


class ModelWrapper(nn.Module):
    def __init__(self, embed_size, word_vocab_size, type_vocab_size, position_vocab_size, num_layers):
        super(ModelWrapper, self).__init__()
        self.bert = Bert(embed_size, word_vocab_size, type_vocab_size, position_vocab_size, num_layers)
        self.out = nn.Linear(embed_size, NUM_CLASSES)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)


    def forward(self, word_ids, token_ids):
        # TODO: Check if attention mask is needed for it
        bert_outputs = self.bert(word_ids, token_ids)
        bert_outputs = self.dropout(bert_outputs)
        logits = self.out(bert_outputs)
        return logits



