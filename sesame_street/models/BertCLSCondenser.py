from torch import nn


class BertCLSCondenser(nn.Module):

    def __init__(self, input_size):
        super(BertCLSCondenser, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.act = nn.Tanh()

    def forward(self, encoder_outputs):
        cls = encoder_outputs[:,0]
        x = self.fc1(cls)
        x = self.act(x)
        return x
