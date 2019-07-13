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
from torch.autograd import Variable
from prediction import RPPNDataset

from prediction.convlstm import ConvLSTM
from prediction.lstm import LSTM
from prediction.rpp_loss import iou_loss

'''model selection'''
models = ('lstm', 'convlstm')

model = models[0]

if model == 'lstm':
    net = LSTM(5, 5)
elif model == 'convlstm':
    net = ConvLSTM(input_channels=150, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)

net.cuda()

'''optimizer & learning rate'''
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

'''data loader'''
video_files = '/home/wuyang/kitty/testing/seq_list.txt'
dataset = 'kitti'
data_loader = RPPNDataset(video_files, dataset).getDataLoader(shuffle=False)

'''
training process
:param train_x: size: (batch, 150, 5)
:param train_y: size (batch, 30, 5)
'''
total_iter = 0
print_freq = 10
save_freq = 1000
h, w = 375, 1242


def resize(data: torch.Tensor):

    bbox, complexity = data[:, :, :4], data[:, :, 4]

    bbox[:, :, 0] = bbox[:, :, 0] * h
    bbox[:, :, 1] = bbox[:, :, 1] * w
    bbox[:, :, 2] = bbox[:, :, 2] * h
    bbox[:, :, 3] = bbox[:, :, 3] * w
    bbox = bbox.int()

    return bbox, complexity


for epoch in range(100):
    for batch_id, data in enumerate(data_loader):

        total_iter += 1
        iter_start_time = time.time()
        train_x, train_y = data

        var_x = Variable(train_x).cuda()
        var_y = Variable(train_y).cuda()

        optimizer.zero_grad()

        out = net(var_x)

        # resize
        out_bbox, out_complexity = resize(out)
        label_bbox, label_complexity = resize(var_y)

        start = time.time()
        #loss_iou = iou_loss(out_bbox, label_bbox)
        loss_iou = nn.L1Loss(out_bbox, label_bbox)
        #loss_iou = Variable(loss_iou, requires_grad=True)

        loss_iou.backward()
        # loss_complexity.backward()
        optimizer.step()
        if total_iter % print_freq == 0:
            print('Epoch: {}, Batch {}, IoU Loss:{:.5f}'.format(epoch, batch_id + 1, loss_iou.data))

        if total_iter % save_freq == 0:
            torch.save(net.state_dict(), 'rppn_checkpoint.pth')
