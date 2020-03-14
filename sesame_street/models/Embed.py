from torch import nn
import torch
from constants.model_consts import *


class Embed(nn.Module):
    def __init__(self, embed_size, word_vocab_size, type_vocab_size, position_vocab_size):
        super(Embed, self).__init__()
        # TODO: Do they prevent these from getting trained?
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(position_vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, LAYER_NORM_EPS)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

    def forward(self, input_ids, token_ids):
        device = input_ids.device
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).to(device)
        # TODO: Token_type_ids is never None
        x1 = self.word_embeddings(input_ids)
        x2 = self.token_type_embeddings(token_ids)
        # TODO: Why not reset position for every [SEP]
        x3 = self.position_embeddings(position_ids)
        x = x1 + x2 + x3
        # TODO: How did they know that this is required?
        return self.dropout(self.layer_norm(x))







