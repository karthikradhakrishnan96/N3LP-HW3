import torch
from torch import nn

from constants.constants import *
from transformers import BertModel
import torch.nn.functional as F

class HfModelWrapper(nn.Module):
    def __init__(self, bert_name):
        super(HfModelWrapper, self).__init__()
        #self.bert = Bert(EMBED_SIZE, WORD_VOCAB_SIZE, TYPE_VOCAB_SIZE, POS_VOCAB_SIZE, NUM_LAYERS)
        self.bert = BertModel.from_pretrained(bert_name)
        self.mid = nn.Linear(2*EMBED_SIZE, 512)
        self.out = nn.Linear(512, NUM_CLASSES)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT)


    def forward(self, input_ids, token_ids, mask_ids = None, input_ids2 = None, token_ids2 = None, mask_ids2 = None):
        _, bert_outputs = self.bert(input_ids, attention_mask = mask_ids, token_type_ids = token_ids)
        _, bert_outputs2 = self.bert(input_ids2, attention_mask = mask_ids2, token_type_ids = token_ids2)
        bert_outputs = torch.cat([bert_outputs, bert_outputs2], dim = 1)

        bert_outputs = self.dropout(bert_outputs)
        bert_outputs = F.relu(self.mid(bert_outputs))
        logits = self.out(bert_outputs)
        return logits
