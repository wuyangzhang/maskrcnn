from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # self.reg = nn.Linear(150, 30)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # designed for #2 input shape : batch_size, seq_length (total frame number), 5 features * 30 bbox/frame. Designed for LSTM input
        self.reg = nn.Linear(5, 1)

        self.reg2 = nn.Linear(self.hidden_size, 160)

    def forward(self, x):
        x, _ = self.rnn(x)

        # designed for #1 input shape : batch_size, seq_length (total bbox number), 5 features

        x = x.permute(0, 2, 1).contiguous()
        x = self.reg(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.relu(x)
        # designed for #2 input shape : batch_size, seq_length (total frame number), 5 features * 30 bbox/frame. Designed for LSTM input
        x = self.reg2(x)
        x = self.sigmoid(x)
        x = x.reshape(x.shape[0], x.shape[2] // 5, 5)
        return x
