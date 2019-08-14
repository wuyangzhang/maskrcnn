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

import sys
sys.path.append('/home/nvidia/maskrcnn')

from prediction import RPPNDataset

from prediction.lstm_single import LSTM

from prediction.preprocesing1 import reorder
from config import Config


'''model selection'''
models = ('lstm', 'convlstm')

model = models[0]
config = Config()

# should convert the input shape to (batch_size, length(5), num of features(30*5))
net = None
if model == 'lstm':
    net = LSTM(input_size=4, hidden_size=16, window=config.window_size, num_layers=2).cuda()
    net.load_state_dict(torch.load(config.model_path))

net.train()
'''optimizer & learning rate'''
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

'''data loader'''
train_video_files = config.home_addr + 'kitty/training/seq_list.txt'
dataset = 'kitti'
train_data = RPPNDataset(train_video_files, dataset)
train_data_loader = train_data.getDataLoader(batch_size=32, window_size=config.window_size, shuffle=True)
shape = train_data.shape
test_video_files = config.home_addr + 'kitty/testing/seq_list.txt'

eval_data_loader = RPPNDataset(test_video_files, dataset).getDataLoader(batch_size=32, window_size=config.window_size, shuffle=False)

'''
training process
:param train_x: size: (batch, 150, 5)
:param train_y: size (batch, 30, 5)
'''
total_iter = 0
print_freq = 10
eval_freq = 10
save_freq = 1
h, w = 375, 1242
max_bbox_num = 32
complexity_loss_weight = 1
mse = nn.MSELoss()

for epoch in range(100000):

    for batch_id, data in enumerate(train_data_loader):

        train_x, train_y, _ = data
        train_x = train_x.cuda()

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 5)
        train_x = train_x[:, :, :, :4]

        train_x = reorder(train_x)
        x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 4)
        x = x.permute(0, 2, 1, 3)
        nonpad = x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2] + x[:, :, :, 3] != 0
        x = x[nonpad].reshape(-1, config.window_size, 4)
        labels = x[:, -1, :]
        x, loss = net(x, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % print_freq == 0:
            print('Epoch: {}, batch {}, '
                  'IoU loss:{:.5f}, '
                  'score loss:{:.5f}, '
                  'cc loss:{:.5f}'.format(
                epoch+1, batch_id + 1,
                loss.item(),
                loss.item(),loss.item()))
                #  loss_iou.item(),
                #  loss_score.item(),
                #  loss_complexity.item()))
            print(net.metrics)

        if epoch % save_freq == 0:
            torch.save(net.state_dict(), 'models/lstm_single_checkpoint{}.pth'.format(epoch))
    # if epoch % eval_freq == 0:
    #     # evaluate the model.
    #     net.eval()
    #     net.load_state_dict(torch.load(config.model_path))
    #     ious = []
    #     inference_time = []
    #     with torch.no_grad():
    #         iou_losses, complexity_losses = [], []
    #         for batch_id, data in enumerate(eval_data_loader):
    #             train_x, train_y, _ = data
    #             train_x = train_x.cuda()
    #             train_y = train_y.cuda()
    #
    #             train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 5)
    #             train_x = train_x[:, :, :, :4]
    #
    #             train_x = reorder(train_x)
    #             x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 4)
    #             x = x.permute(0, 2, 1, 3)
    #             nonpad = x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2] + x[:, :, :, 3] != 0
    #             x = x[nonpad].reshape(-1, config.window_size, 4)
    #             labels = x[:, -1, :]
    #             if len(x) == 0:
    #                 continue
    #             s = time.time()
    #             out = net(x)
    #             inference_time.append(time.time()-s)
    #
    #             # calculate inference time.
    #             target = labels
    #             nonpad = target[:, 0] + target[:, 1] + target[:, 2] + target[:, 3] != 0
    #             target = target[nonpad]
    #             # x, target shape : batch, 32, 4
    #             x = out
    #             x1 = torch.max(x[:, 0], target[:, 0])
    #             y1 = torch.max(x[:, 1], target[:, 1])
    #             x2 = torch.min(x[:, 2], target[:, 2])
    #             y2 = torch.min(x[:, 3], target[:, 3])
    #
    #             inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    #             # pred_area = torch.max((x[:, :, 2] - x[:, :, 0]) * (x[:, :, 3] - x[:, :, 1]), torch.tensor([0.]).cuda())
    #             pred_area = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    #
    #             target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    #             # union = torch.max(pred_area + target_area - inter_area, torch.tensor([0.]).cuda())
    #             union = pred_area + target_area - inter_area
    #             iou = inter_area / (torch.tensor([1e-6]).cuda() + union)
    #             ious.append(iou.mean().item())
    #
    #             #find iou loss
    #
    #         print('avg inference time {}, avg iou {}'.format(
    #             sum(inference_time) / len(inference_time),
    #             sum(ious) / len(ious)
    #         ))
