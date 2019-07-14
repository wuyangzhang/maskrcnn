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
save_freq = 1000
h, w = 375, 1242

for epoch in range(100):
    for batch_id, data in enumerate(train_data_loader):

        total_iter += 1
        iter_start_time = time.time()
        train_x, train_y = data
        train_x = train_x.cuda()
        train_y = train_y.cuda()

        out = net(train_x)
        optimizer.zero_grad()

        out_bbox = out[:, :, :4]
        label_bbox = train_y[:, :, :4]

        loss_iou = iou_loss(out_bbox, label_bbox)

        loss_iou.backward()
        # loss_complexity.backward()
        optimizer.step()
        if total_iter % print_freq == 0:
            print('Epoch: {}, Batch {}, IoU Loss:{:.5f}'.format(epoch, batch_id + 1, loss_iou.data))

        if total_iter % save_freq == 0:
            torch.save(net.state_dict(), 'rppn_checkpoint.pth')


def resize(data: torch.Tensor):
    bbox, complexity = data[:, :, :4], data[:, :, 4]

    bbox[:, :, 0] = bbox[:, :, 0] * h
    bbox[:, :, 1] = bbox[:, :, 1] * w
    bbox[:, :, 2] = bbox[:, :, 2] * h
    bbox[:, :, 3] = bbox[:, :, 3] * w
    bbox = bbox.int()
    return bbox, complexity
