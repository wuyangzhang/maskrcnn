import time

import torch

from prediction.convlstm import ConvLSTM
from prediction.lstm_single import LSTM
from prediction.preprocesing1 import reorder
import logging

#log = logging.getLogger('maskrcnn')


class PredictionDelegator:
    def __init__(self, config):
        self.config = config
        if config.pred_algo == 0:
            self.model = LSTM(input_size=4, hidden_size=16, window=config.window_size, num_layers=2)
            self.model.load_state_dict(torch.load(self.config.model_path))

        elif config.pred_algo == 1:
            self.model = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)

        self.model = self.model.eval()
        self.model = self.model.cuda()

    def run(self, input):
        s = time.time()
        input = self.prepare_input(input)
        s1 = time.time()
        print('[prediction] prepare,', s1-s)
        output = self.model(input)
        s = time.time()
        self.config.log.info('[prediction] inference,{}'.format(s-s1))
        res = self.post_proc(output)
        s1 = time.time()
        print('[prediction] post,', s1 - s)
        return res

    def nms_filter(self, bbox):
        return

    def prepare_input(self, queue):

        bbox_list = list()
        for bbox in queue:
            bbox = bbox.bbox
            pad = torch.nn.ConstantPad2d((0, 0, 0, self.config.padding - bbox.shape[0]), 0.)
            bbox = pad(bbox)
            bbox_list.append(bbox)

        # stack all bbox
        bbox_list = torch.stack(bbox_list)
        # normalize
        bbox_list[:, :, 0] = bbox_list[:, :, 0] / self.config.frame_width
        bbox_list[:, :, 1] = bbox_list[:, :, 1] / self.config.frame_height
        bbox_list[:, :, 2] = bbox_list[:, :, 2] / self.config.frame_width
        bbox_list[:, :, 3] = bbox_list[:, :, 3] / self.config.frame_height

        # unsqueeze the first dim
        bbox_list = bbox_list.unsqueeze(dim=0).cuda()

        # key step!. reorder.
        bbox_list = reorder(bbox_list)
        x = bbox_list.reshape(bbox_list.shape[0], bbox_list.shape[1], -1, 4)
        x = x.permute(0, 2, 1, 3)
        nonpad = x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2] + x[:, :, :, 3] != 0
        x = x[nonpad].reshape(-1, self.config.window_size, 4)
        return x

    def post_proc(self, output):
        # convert the normalized prediction results to the real image size.
        output[:, 0] = output[:, 0] * self.config.frame_width
        output[:, 1] = output[:, 1] * self.config.frame_height
        output[:, 2] = output[:, 2] * self.config.frame_width
        output[:, 3] = output[:, 3] * self.config.frame_height
        return output
