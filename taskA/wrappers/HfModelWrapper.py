import torch
from torch import nn
import torch.nn.init as init
from constants.constants import *
from transformers import BertModel
import torch.nn.functional as F


class HfModelWrapper(nn.Module):
    def __init__(self, bert_name):
        super(HfModelWrapper, self).__init__()
        # self.bert = Bert(EMBED_SIZE, WORD_VOCAB_SIZE, TYPE_VOCAB_SIZE, POS_VOCAB_SIZE, NUM_LAYERS)
        self.bert = BertModel.from_pretrained(bert_name)
        self.feats_len = 34
        self.mid = nn.Linear(2 * EMBED_SIZE + self.feats_len, 512)
        self.mid2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, NUM_CLASSES)
        self.dropout2 = nn.Dropout(0.17)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT)
        init.kaiming_normal_(self.mid.weight, mode='fan_in')

        init.kaiming_normal_(self.out.weight, mode='fan_in')
        init.kaiming_normal_(self.mid2.weight, mode='fan_in')

    def forward(self, input_ids, token_ids, mask_ids=None, input_ids2=None, token_ids2=None, mask_ids2=None,
                feats=None):
        _, bert_outputs = self.bert(input_ids, attention_mask=mask_ids, token_type_ids=token_ids)
        _, bert_outputs2 = self.bert(input_ids2, attention_mask=mask_ids2, token_type_ids=token_ids2)

        bert_outputs = torch.cat([bert_outputs, bert_outputs2], dim=1)
        if feats is not None:
            bert_outputs = torch.cat([bert_outputs, feats], dim=1)
        bert_outputs = self.dropout(bert_outputs)
        bert_outputs = self.dropout2(F.relu(self.mid(bert_outputs)))
        bert_outputs = F.relu(self.mid2(bert_outputs))
        logits = self.out(bert_outputs)
        return logits
