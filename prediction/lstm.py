from torch import nn
import torch



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window, num_layers=4):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.relu = nn.ReLU()
        self.reg = nn.Linear(window, 1)
        self.score_lin = nn.Linear(self.hidden_size, 32)
        self.reg2 = nn.Linear(self.hidden_size, 128)
        self.metrics = {}
        self.mse_loss = nn.MSELoss()

    def forward(self, x, target=None):

        x, h_state = self.rnn(x)
        x = x[:, -1, :]
        x = self.relu(x)
        score = self.score_lin(x)
        score = torch.sigmoid(score)
        score = score#.unsqueeze(-1)
        x = self.reg2(x)
        x = torch.sigmoid(x)
        x = x.reshape(x.shape[0], -1, 4)
        if target is None:
            return x, score


        # x, target shape : batch, 32, 4
        x1 = torch.max(x[:, :, 0], target[:, :, 0])
        y1 = torch.max(x[:, :, 1], target[:, :, 1])
        x2 = torch.min(x[:, :, 2], target[:, :, 2])
        y2 = torch.min(x[:, :, 3], target[:, :, 3])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        #pred_area = torch.max((x[:, :, 2] - x[:, :, 0]) * (x[:, :, 3] - x[:, :, 1]), torch.tensor([0.]).cuda())
        pred_area = (x[:, :, 2] - x[:, :, 0]) * (x[:, :, 3] - x[:, :, 1])

        target_area = (target[:, :, 2] - target[:, :, 0]) * (target[:, :, 3] - target[:, :, 1])
        #union = torch.max(pred_area + target_area - inter_area, torch.tensor([0.]).cuda())
        union = pred_area + target_area - inter_area
        iou = inter_area / (torch.tensor([1e-16]).cuda() + union)
        iou_loss = torch.mean(1 - iou)
        # loss_x1 = self.mse_loss(x[:, :, 0], target[:, :, 0])
        # loss_y1 = self.mse_loss(x[:, :, 1], target[:, :, 1])
        # loss_x2 = self.mse_loss(x[:, :, 2], target[:, :, 2])
        # loss_y2 = self.mse_loss(x[:, :, 3], target[:, :, 3])

        nonpad_mask = target[:, :, 0] + target[:, :, 1] + target[:, :, 2] + target[:, :, 3] != 0
        nonpad_mask = nonpad_mask
        loss_x1 = self.mse_loss(x[:, :, 0][nonpad_mask], target[:, :, 0][nonpad_mask])
        loss_y1 = self.mse_loss(x[:, :, 1][nonpad_mask], target[:, :, 1][nonpad_mask])
        loss_x2 = self.mse_loss(x[:, :, 2][nonpad_mask], target[:, :, 2][nonpad_mask])
        loss_y2 = self.mse_loss(x[:, :, 3][nonpad_mask], target[:, :, 3][nonpad_mask])

        total_loss = loss_x1 + loss_x2 + loss_y1 + loss_y2 #+ iou_loss

        self.metrics = {
            'loss': total_loss.item(),
            'x1' : loss_x1.item(),
            'y1' : loss_y1.item(),
            'x2' : loss_x2.item(),
            'y2' : loss_y2.item()
        }

        return x, score, total_loss

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, window, num_layers=4):
#         super(LSTM, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.center_shift_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#         self.pos_shift_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#
#         self.relu = nn.ReLU()
#         self.reg = nn.Linear(window, 1)
#         self.score_lin = nn.Linear(4, 1)
#
#         self.center_reg = nn.Linear(self.hidden_size, 64)
#         self.pos_reg = nn.Linear(self.hidden_size, 64)
#
#     def forward(self, x):
#         # x input: x (batch, time, 128)
#         # left up cornet x0, y0: shape (batch, time, 64)
#         x = x.reshape(x.shape[0], x.shape[1], -1, 4)
#
#         coord = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2).cuda()
#         coord[:, :, :, 0], coord[:, :, :, 1] = x[:, :, :, 0], x[:, :, :, 1]
#         coord = coord.reshape(coord.shape[0], coord.shape[1], -1)
#
#         # size x0, y0, shape (batch, time, 64)
#         size = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2).cuda()
#         size[:, :, :, 0], size[:, :, :, 1] = x[:, :, :, 2] - x[:, :, :, 0], x[:, :, :, 3] - x[:, :, :, 1]
#         size = size.reshape(size.shape[0], size.shape[1], -1)
#
#         coord, _ = self.center_shift_rnn(coord)
#         coord = coord[:, -1, :].squeeze(1)
#         coord = self.relu(coord)
#         coord = self.center_reg(coord)
#         coord = torch.sigmoid(coord)
#         coord = coord.reshape(coord.shape[0], -1, 2)
#
#         size, _ = self.pos_shift_rnn(size)
#         size = self.pos_reg(size)
#         size = size[:, -1, :].squeeze(1)
#         size = self.relu(size)
#         size = torch.sigmoid(size)
#         size = size.reshape(size.shape[0], -1, 2)
#
#         # generate new outputs.
#         output = torch.zeros(size.shape[0], size.shape[1], 4).cuda()
#         output[:, :, 0], output[:, :, 1] = coord[:, :, 0], coord[:, :, 1]
#         output[:, :, 2] = coord[:, :, 0] + size[:, :, 0]
#         output[:, :, 3] = coord[:, :, 1] + size[:, :, 1]
#         output = output.reshape(output.shape[0], output.shape[1], -1)
#
#         score = self.score_lin(output)
#         score = torch.sigmoid(score)
#         return output, score
