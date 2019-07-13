from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=4):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(150, 30)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.rnn(x)
        # s, b, h = x.shape
        # x = x.view(s, b * h)
        x = x.permute(0, 2, 1).contiguous()
        x = self.reg(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.relu(x)
        x = self.sigmoid(x)
        return x