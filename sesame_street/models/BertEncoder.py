from torch import nn
from sesame_street.constants.model_consts import *

class BertEncoder(nn.Module):
    def __init__(self, num_layers, input_size):
        super(BertEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_size, NUM_HEADS, dim_feedforward=ENCODER_FF_DIM,
                                                   activation="gelu")
        self.encoder_stack = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, embeds, mask_ids = None):
        if mask_ids is not None:
          mask_ids = mask_ids == 0
        x = self.encoder_stack(embeds.permute(1, 0, 2), src_key_padding_mask = mask_ids)
        return x.permute(1,0,2)
