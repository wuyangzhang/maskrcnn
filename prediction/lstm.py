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

# net = LSTM(5, 5)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#
#
# video_files = '/home/wuyang/kitty/testing/seq_list.txt'
# dataset = 'kitti'
# data_loader = RPPNDataset(video_files, dataset).getDataLoader(shuffle=False)
#
#
# '''
# training process
# :param train_x: size: (batch, 150, 5)
# :param train_y: size (batch, 30, 5)
# '''
# total_iter = 0
# print_freq = 10
# save_freq = 1000
#
# for epoch in range(100):
#     for batch_id, data in enumerate(data_loader):
#
#         iter_start_time = time.time()
#         train_x, train_y = data
#
#         var_x = Variable(train_x)
#         var_y = Variable(train_y)
#
#         out = net(var_x)
#         loss = iou_loss(out, var_y)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if total_iter  % print_freq == 0:
#             print('Epoch: {}, Batch {}, Loss:{:.5f}'.format(epoch, batch_id + 1, loss.data))
#
#         if total_iter % save_freq == 0:
#             torch.save(net.state_dict(), 'rppn_checkpoint.pth')
