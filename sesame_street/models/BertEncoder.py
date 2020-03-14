from torch import nn
from constants.model_consts import *


class BertEncoder(nn.Module):
    def __init__(self, num_layers, input_size):
        super(BertEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_size, NUM_HEADS, dim_feedforward=ENCODER_FF_DIM,
                                                   activation="gelu")
        self.encoder_stack = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, embeds):
        x = self.encoder_stack(embeds)
        return x
