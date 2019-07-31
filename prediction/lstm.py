from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window, num_layers=4):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # self.reg = nn.Linear(150, 30)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

        # designed for #2 input shape : batch_size, seq_length (total frame number), 5 features * 30 bbox/frame. Designed for LSTM input
        #self.reg = nn.Linear(window * hidden_size, hidden_size)
        self.reg = nn.Linear(window, 1)
        self.score_lin = nn.Linear(5, 1)
        #self.score_sigmoid = nn.Sigmoid()
        self.reg2 = nn.Linear(self.hidden_size, 160)

    def forward(self, x):
        x, _ = self.rnn(x)

        # designed for #1 input shape : batch_size, seq_length (total bbox number), 5 features

        x = x.permute(0, 2, 1).contiguous()
        x = self.reg(x)
        x = x.permute(0, 2, 1).contiguous()

        #x = x.reshape(x.shape[0], -1)
        x = self.relu(x)
        x = self.reg2(x)
        #x = x.reshape(x.shape[0], -1, 5)
        #score = self.score_lin(x)
        #score = score.reshape(score.shape[0], -1)
        #score = self.score_sigmoid(score)
        # designed for #2 input shape : batch_size, seq_length (total frame number), 5 features * 30 bbox/frame. Designed for LSTM input
        #x = self.reg2(x)
        x = torch.sigmoid(x)

        x = x.reshape(x.shape[0], -1, 5)
        score = self.score_lin(x)
        score = torch.sigmoid(score)
        return x, score
