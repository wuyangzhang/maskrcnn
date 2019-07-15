import torch

from prediction.convlstm import ConvLSTM
from prediction.lstm import LSTM


class PredictionDelegator:
    def __init__(self, config):
        self.config = config
        if config.pred_algo == 0:
            self.model = LSTM(input_size=160, hidden_size=64, num_layers=4)
            self.model.load_state_dict(torch.load(self.config.model_path))
            self.model.eval()
        elif config.pred_algo == 1:
            self.model = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)
        self.model.cuda()


    def run(self, input):
        input = self.prepare_input(input)
        input = input.cuda()
        output = self.model(input)
        self.post_proc(output)

    def prepare_input(self, queue):
        # queue , bboxes, complexity
        res = []
        for bboxes, overhead in queue:
            bboxes[:, 0] = bboxes[:, 0] / self.config.frame_height
            bboxes[:, 1] = bboxes[:, 1] / self.config.frame_width
            bboxes[:, 2] = bboxes[:, 2] / self.config.frame_height
            bboxes[:, 3] = bboxes[:, 3] / self.config.frame_width
            overhead = overhead / self.config.max_complexity
            bboxes = torch.cat([bboxes, overhead], dim=1)
            pad = torch.nn.ConstantPad2d((0, 0, 0, self.config.padding-bboxes.shape[0]), 0.)
            bboxes = pad(bboxes)
            res.append(bboxes)
        res = torch.stack(res)
        return res.view(1, self.config.max_queue_size, -1)

    def post_proc(self, output):
        output = output.squeeze(0)
        bbox = output[:, :4]
        overhead = output[:, -1]