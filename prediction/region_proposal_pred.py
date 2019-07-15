'''
We predict coordinate distribution and computing complexity
of region proposals in the next frame.
Two problems: how does the network know the number of output?
how to calculate the IOU which requires to correctly map predicted
region proposals to the labels.?
'''

import time
from torch import nn
import torch.optim
from prediction import RPPNDataset

from prediction.convlstm import ConvLSTM
from prediction.lstm import LSTM
from prediction.rpp_loss import iou_loss


class PredictionDelegator:
    def __init__(self, config):
        self.config = config
        if config.pred_algo == 0:
            self.model = LSTM(input_size=160, hidden_size=64, num_layers=4)
        elif config.pred_algo == 1:
            self.model = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)
        self.model.cuda()
        self.model.eval()

    def run(self, input):
        return self.model(input)

'''model selection'''
models = ('lstm', 'convlstm')

model = models[0]

# should convert the input shape to (batch_size, length(5), num of features(30*5))
if model == 'lstm':
    net = LSTM(input_size=160, hidden_size=64, num_layers=4)
elif model == 'convlstm':
    net = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)

# net.train()
net.cuda()

'''optimizer & learning rate'''
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

'''data loader'''
train_video_files = '/home/wuyang/kitty/training/seq_list.txt'
dataset = 'kitti'
train_data_loader = RPPNDataset(train_video_files, dataset).getDataLoader(batch_size=16, shuffle=True)

test_video_files = '/home/wuyang/kitty/testing/seq_list.txt'

eval_data_loader = RPPNDataset(test_video_files, dataset).getDataLoader(batch_size=16, shuffle=True)

'''
training process
:param train_x: size: (batch, 150, 5)
:param train_y: size (batch, 30, 5)
'''
total_iter = 0
print_freq = 10
eval_freq = 1
save_freq = 1
h, w = 375, 1242
max_bbox_num = 32
complexity_loss_weight = 0.1

for epoch in range(100):
    for batch_id, data in enumerate(train_data_loader):

        total_iter += 1
        iter_start_time = time.time()
        train_x, train_y = data
        train_x = train_x.cuda()
        labels = train_y.cuda()

        out = net(train_x)
        optimizer.zero_grad()

        loss_iou, loss_complexity = iou_loss(out, labels, max_bbox_num)
        loss = loss_iou + loss_complexity * complexity_loss_weight
        loss.backward()
        # loss.backward()
        # loss_complexity.backward()
        optimizer.step()
        if total_iter % print_freq == 0:
            print('Epoch: {}, batch {}, IoU loss:{:.5f}, computing complexity loss:{:.5f}'.format(epoch, batch_id + 1,
                                                                                                  loss_iou.item(),
                                                                                                  loss_complexity.item()))

        # if epoch % save_freq == 0:
        #     torch.save(net.state_dict(), 'rppn_checkpoint.pth')

    if epoch % eval_freq == 0:
        # evaluate the model.
        with torch.no_grad():
            iou_losses, complexity_losses = [], []
            for batch_id, data in enumerate(eval_data_loader):
                train_x, train_y = data
                train_x = train_x.cuda()
                train_y = train_y.cuda()
                out = net(train_x)
                loss_iou, loss_complexity = iou_loss(out, train_y, max_bbox_num)
                iou_losses.append(loss_iou)
                complexity_losses.append(loss_complexity)

        print('[Testing] IOU loss {}, computing complexity loss {} over the test dataset.'.
              format(sum(iou_losses) / len(iou_losses), sum(complexity_losses) / len(complexity_losses)))


def resize(data: torch.Tensor):
    bbox, complexity = data[:, :, :4], data[:, :, 4]

    bbox[:, :, 0] = bbox[:, :, 0] * h
    bbox[:, :, 1] = bbox[:, :, 1] * w
    bbox[:, :, 2] = bbox[:, :, 2] * h
    bbox[:, :, 3] = bbox[:, :, 3] * w
    bbox = bbox.int()
    return bbox, complexity
