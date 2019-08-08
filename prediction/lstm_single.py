from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window, num_layers=4):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.reg = nn.Linear(window, 1)
        self.score_lin = nn.Linear(self.hidden_size, 1)
        self.reg2 = nn.Linear(self.hidden_size * 2, 4)
        self.metrics = {}
        self.mse_loss = nn.MSELoss()
        self.tanh = torch.nn.Sigmoid()  # torch.nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, x, target=None):
        x, h_state = self.rnn(x)
        x = x[:, -1, :]
        x = self.relu(x)
        # score = self.score_lin(x)
        # score = torch.sigmoid(score)
        # score = score  # .unsqueeze(-1)
        x = self.reg2(x)
        x = self.tanh(x)

        if target is None:
            return x

        target = target.squeeze()
        nonpad = target[:, 0] + target[:, 1] + target[:, 2] + target[:, 3] != 0
        target = target[nonpad]
        # x, target shape : batch, 32, 4
        x1 = torch.max(x[:, 0], target[:, 0])
        y1 = torch.max(x[:, 1], target[:, 1])
        x2 = torch.min(x[:, 2], target[:, 2])
        y2 = torch.min(x[:, 3], target[:, 3])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        # pred_area = torch.max((x[:, :, 2] - x[:, :, 0]) * (x[:, :, 3] - x[:, :, 1]), torch.tensor([0.]).cuda())
        pred_area = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])

        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        # union = torch.max(pred_area + target_area - inter_area, torch.tensor([0.]).cuda())
        union = pred_area + target_area - inter_area
        iou = inter_area / (torch.tensor([1e-6]).cuda() + union)
        iou_loss = torch.mean(1 - iou)

        loss_x1 = self.mse_loss(x[:, 0], target[:, 0])
        loss_y1 = self.mse_loss(x[:, 1], target[:, 1])
        loss_x2 = self.mse_loss(x[:, 2], target[:, 2])
        loss_y2 = self.mse_loss(x[:, 3], target[:, 3])

        total_loss = loss_x1 + loss_x2 + loss_y1 + loss_y2 + iou_loss

        self.metrics = {
            'loss': total_loss.item(),
            'iou': iou_loss.item(),
            'x1': loss_x1.item(),
            'y1': loss_y1.item(),
            'x2': loss_x2.item(),
            'y2': loss_y2.item()
        }

        return x, total_loss
