import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from prediction import RPPNDataset
from prediction.rpp_loss import iou_loss
from prediction.nms import nms
from config import Config

batch_size = 32


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size).cuda()

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, window_size=5):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.dropout_p = dropout_p

        # set the attention upon different region proposals
        self.attn = nn.Linear(self.hidden_size, self.hidden_size // 5)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size, self.hidden_size, batch_first=True).cuda()

        self.reg = nn.Linear(5, 1)
        self.out = nn.Linear(self.hidden_size, 160)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.window = window_size

    def forward(self, encoder_outputs, hidden):
        input = self.dropout(encoder_outputs)

        # calculate attention
        attn = self.attn(input)
        attn = attn.view(attn.shape[0], -1)
        attn_weights = F.softmax(attn, dim=1)
        attn_weights = attn_weights.view(attn.shape[0], self.window, 1, -1)

        attn_weights = attn_weights.repeat(1, 1, 5, 1)
        attn_weights = attn_weights.transpose(2, 3)
        attn_weights = attn_weights.reshape(attn.shape[0], self.window, -1)

        # apply attention
        attn_applied = torch.mul(input, attn_weights)

        output, hidden = self.gru(attn_applied)
        output = output.permute(0, 2, 1).contiguous()
        output = self.reg(output)
        output = output.permute(0, 2, 1).contiguous()
        output = self.relu(output)
        output = self.out(output)
        output = self.sigmoid(output)
        output = output.reshape(output.shape[0], output.shape[2] // 5, 5)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()



def train():
    config = Config()

    encoder = EncoderRNN(160, 160).cuda()
    encoder.load_state_dict(torch.load('en_rppn_checkpoint.pth'))
    decoder = AttnDecoderRNN(160, 160, window_size=config.window_size).cuda()
    decoder.load_state_dict(torch.load('de_rppn_checkpoint.pth'))
    config = Config()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    train_video_files = config.home_addr + 'kitty/training/seq_list.txt'
    dataset = 'kitti'
    train_data = RPPNDataset(train_video_files, dataset)
    train_data_loader = train_data.getDataLoader(batch_size=batch_size,
                                                 window_size=config.window_size,
                                                 shuffle=True)
    shape = train_data.shape
    test_video_files = config.home_addr + 'kitty/testing/seq_list.txt'

    eval_data_loader = RPPNDataset(test_video_files, dataset).getDataLoader(batch_size=batch_size,
                                                                            window_size=config.window_size,
                                                                            shuffle=True)

    total_iter = 0
    max_bbox_num = 32
    complexity_loss_weight = 0.1

    for epoch in range(100):
        for batch_id, data in enumerate(train_data_loader):
            # break
            encoder_hidden = encoder.initHidden()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            train_x, train_y, _ = data
            train_x = train_x.cuda()
            labels = train_y.cuda()

            input_length = train_x.size(1)
            encoder_outputs = list()

            # iterate the last K timestamps
            for ei in range(input_length):
                # get the tensor of a single timestamp
                input = train_x[:, ei, :].reshape(train_x.size(0), 1, train_x.size(2))
                encoder_output, encoder_hidden = encoder(input, encoder_hidden)
                encoder_outputs.append(encoder_output)

            # cat all the outputs from the encoder
            encoder_outputs = torch.cat(encoder_outputs, dim=1)

            # the decoder
            decoder_output, _, decoder_attention = decoder(encoder_outputs, encoder_hidden)

            decoder_output = nms(decoder_output, shape)

            loss_iou, loss_complexity = iou_loss(decoder_output, labels, max_bbox_num)
            loss = loss_iou
            # loss = loss_iou + loss_complexity * complexity_loss_weight
            loss.backward(retain_graph=True)

            encoder_optimizer.step()
            decoder_optimizer.step()

            if batch_id % 10 == 0:
                print(
                    'Epoch: {}, batch {}, IoU loss:{:.5f}, cc loss:{:.5f}'.format(epoch + 1, batch_id + 1,
                                                                                  loss_iou.item(),
                                                                                  loss_complexity.item()))

        if epoch % 1 == 0:
            torch.save(encoder.state_dict(), 'en_rppn_checkpoint.pth')
            torch.save(decoder.state_dict(), 'de_rppn_checkpoint.pth')

        # with torch.no_grad():
        #     iou_losses, complexity_losses = [], []
        #     for batch_id, data in enumerate(eval_data_loader):
        #         if batch_id > 692:
        #             break
        #         train_x, train_y, _ = data
        #         # print(batch_id, train_x.shape)
        #         train_x = train_x.cuda()
        #         train_y = train_y.cuda()
        #         encoder_outputs = list()
        #
        #         input_length = train_x.size(1)
        #
        #         # iterate the last K timestamps
        #         for ei in range(input_length):
        #             # get the tensor of a single timestamp
        #             input = train_x[:, ei, :].reshape(train_x.size(0), 1, train_x.size(2))
        #             encoder_output, encoder_hidden = encoder(input, encoder_hidden)
        #             encoder_outputs.append(encoder_output)
        #
        #         # cat all the outputs from the encoder
        #         encoder_outputs = torch.cat(encoder_outputs, dim=1)
        #
        #         # the decoder
        #         decoder_output, _, decoder_attention = decoder(encoder_outputs, encoder_hidden)
        #
        #         # apply nms
        #         decoder_output = nms(decoder_output, shape)
        #
        #         loss_iou, loss_complexity = iou_loss(decoder_output, train_y, max_bbox_num)
        #         iou_losses.append(loss_iou)
        #         complexity_losses.append(loss_complexity)
        #
        # print('[Testing] IOU loss {}, computing complexity loss {} over the test dataset.'.
        #       format(sum(iou_losses) / len(iou_losses), sum(complexity_losses) / len(complexity_losses)))


if __name__ == "__main__":
    train()
