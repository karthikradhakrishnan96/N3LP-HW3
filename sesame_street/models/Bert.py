from torch import nn
from sesame_street.models.BertCLSCondenser import BertCLSCondenser
from sesame_street.models.BertEncoder import BertEncoder
from sesame_street.models.Embed import Embed
import torch


class Bert(nn.Module):
    def __init__(self, embed_size, word_vocab_size, type_vocab_size, position_vocab_size, num_layers):
        super(Bert, self).__init__()
        self.embedder = Embed(embed_size, word_vocab_size, type_vocab_size, position_vocab_size)
        self.encoder = BertEncoder(num_layers, embed_size)
        self.classifier = BertCLSCondenser(embed_size)

    def forward(self, input_ids, token_ids):
        embeddings = self.embedder.forward(input_ids, token_ids)
        encoder_outputs = self.encoder.forward(embeddings)
        classifier_outputs = self.classifier(encoder_outputs)
        return classifier_outputs

    @staticmethod
    def from_hugging_face(pre_trained_weights_path):
        weights = torch.load(pre_trained_weights_path)
        print("hello")




if __name__ == "__main__":
    bert = Bert.from_hugging_face("./saved_models/bert_base_uncased.bin")

