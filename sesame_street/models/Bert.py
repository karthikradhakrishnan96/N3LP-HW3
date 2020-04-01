from torch import nn
from sesame_street.models.BertCLSCondenser import BertCLSCondenser
from sesame_street.models.BertEncoder import BertEncoder
from sesame_street.models.Embed import Embed
import torch
import os
from sesame_street.utils.BertWeightLoader import *

class Bert(nn.Module):
    def __init__(self, embed_size, word_vocab_size, type_vocab_size, position_vocab_size, num_layers):
        super(Bert, self).__init__()
        self.embedder = Embed(embed_size, word_vocab_size, type_vocab_size, position_vocab_size)
        self.encoder = BertEncoder(num_layers, embed_size)
        self.condenser = BertCLSCondenser(embed_size)

    def forward(self, input_ids, token_ids, mask_ids = None):
        embeddings = self.embedder.forward(input_ids, token_ids)
        encoder_outputs = self.encoder.forward(embeddings, mask_ids)
        classifier_outputs = self.condenser(encoder_outputs)
        return classifier_outputs


if __name__ == "__main__":
    weights = torch.load(os.path.sep.join([".", "saved_models", "bert-base-uncased-pytorch_model.bin"]))
    model = Bert(768, 30522, 2, 512, 12)
    BertWeightLoader.from_hugging_face(model, weights)
    print("ok")
