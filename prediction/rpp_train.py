'''
We predict coordinate distribution and computing complexity
of region proposals in the next frame.
Two problems: how does the network know the number of output?
how to calculate the IOU which requires to correctly map predicted
region proposals to the labels.?
'''
import time
import torch.nn as nn
import torch.optim
from prediction import RPPNDataset

from prediction.convlstm import ConvLSTM
from prediction.lstm import LSTM
from prediction.rpp_loss import iou_loss
from prediction.nms import nms
from prediction.preprocesing1 import reorder, remove_tiny_bbox
from config import Config


class PredictionDelegator:
    def __init__(self, config):
        self.config = config
        if config.pred_algo == 0:
            self.model = LSTM(input_size=160, hidden_size=64, window=config.window_size, num_layers=4)
        elif config.pred_algo == 1:
            self.model = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)
        self.model.cuda()
        self.model.eval()

    def run(self, input):
        return self.model(input)


'''model selection'''
models = ('lstm', 'convlstm')

model = models[0]
config = Config()

# should convert the input shape to (batch_size, length(5), num of features(30*5))
net = None
if model == 'lstm':
    net = LSTM(input_size=128, hidden_size=64, window=config.window_size, num_layers=2).cuda()
    #net.load_state_dict(torch.load(config.model_path))

elif model == 'convlstm':
    net = ConvLSTM(input_channels=1, hidden_channels=[32, 1], window_size=config.window_size, kernel_size=3).cuda()
    #net.load_state_dict(torch.load(config.model_path))

'''optimizer & learning rate'''
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

'''data loader'''
train_video_files = config.home_addr + 'kitty/training/seq_list.txt'
dataset = 'kitti'
train_data = RPPNDataset(train_video_files, dataset)
train_data_loader = train_data.getDataLoader(batch_size=1, window_size=config.window_size, shuffle=False)
shape = train_data.shape
test_video_files = config.home_addr + 'kitty/testing/seq_list.txt'

eval_data_loader = RPPNDataset(test_video_files, dataset).getDataLoader(batch_size=1, window_size=config.window_size, shuffle=False)

'''
training process
:param train_x: size: (batch, 150, 5)
:param train_y: size (batch, 30, 5)
'''
total_iter = 0
print_freq = 10
eval_freq = 10
save_freq = 5
h, w = 375, 1242
max_bbox_num = 32
complexity_loss_weight = 0.1

mse = nn.MSELoss()

for epoch in range(100000):

    h_state = None
    for batch_id, data in enumerate(train_data_loader):

        total_iter += 1
        iter_start_time = time.time()
        train_x, train_y, _ = data

        train_x = train_x.cuda()
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 5)
        train_x = train_x[:, :, :, :4].reshape(train_x.shape[0], train_x.shape[1], -1)

        train_x = reorder(train_x)

        # labels = train_y.cuda()
        # labels = remove_tiny_bbox(labels)
        labels = train_x[:, -1, :].reshape(train_x.shape[0], -1, 4)
        if model == 'convlstm':
            train_x = train_x.reshape(train_x.shape[0], config.window_size, -1, 5)
            train_x = train_x.unsqueeze(1)
            train_x[:, :, :, :, 4] = 0
        out, h_state = net(train_x, h_state)

        #loss = mse(out[0], labels)
        #out = nms(out, shape)

        loss_iou, loss_score, loss_complexity = iou_loss(out, labels, max_bbox_num)

        loss = loss_iou + loss_score

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # loss.backward()
        # loss_complexity.backward()
        optimizer.step()
        if total_iter % print_freq == 0:
            print('Epoch: {}, batch {}, '
                  'IoU loss:{:.5f}, '
                  'score loss:{:.5f}, '
                  'cc loss:{:.5f}'.format(
                epoch+1, batch_id + 1,
                loss.item(),
                loss.item(),loss.item()))
                # loss_iou.item(),
                # loss_score.item(),
                # loss_complexity.item()))

        if epoch % save_freq == 0:
            torch.save(net.state_dict(), 'lstm_checkpoint{}.pth'.format(epoch))


