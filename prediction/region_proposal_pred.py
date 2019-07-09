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
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

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

for epoch in range(100):
    for batch_id, data in enumerate(data_loader):

        iter_start_time = time.time()
        train_x, train_y = data

        var_x = Variable(train_x).cuda()
        var_y = Variable(train_y).cuda()

        out = net(var_x)
        loss = iou_loss(out, var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if total_iter  % print_freq == 0:
            print('Epoch: {}, Batch {}, Loss:{:.5f}'.format(epoch, batch_id + 1, loss.data))

        if total_iter % save_freq == 0:
            torch.save(net.state_dict(), 'rppn_checkpoint.pth')


