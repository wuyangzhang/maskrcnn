from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window, num_layers=4):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        self.relu = nn.ReLU()


        self.reg = nn.Linear(window, 1)
        self.score_lin = nn.Linear(4, 1)
        self.reg2 = nn.Linear(self.hidden_size, 128)

    def forward(self, x):
        x, _ = self.rnn(x)

        x = x[:, -1:, :].squeeze(1)
        x = self.relu(x)
        x = self.reg2(x)

        x = torch.sigmoid(x)

        x = x.reshape(x.shape[0], -1, 4)
        score = self.score_lin(x[:, :, :4])
        score = torch.sigmoid(score)
        return x, score
