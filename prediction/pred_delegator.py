import torch

from prediction.convlstm import ConvLSTM
from prediction.lstm import LSTM


class PredictionDelegator:
    def __init__(self, config):
        self.config = config
        if config.pred_algo == 0:
            self.model = LSTM(input_size=160, hidden_size=64, window=config.window_size, num_layers=4)
            self.model.load_state_dict(torch.load(self.config.model_path))
            self.model.eval()
        elif config.pred_algo == 1:
            self.model = ConvLSTM(input_channels=30, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)
        self.model.cuda()

    def run(self, input):
        input = self.prepare_input(input)
        input = input.cuda()
        output = self.model(input)
        return self.post_proc(output)

    def nms_filter(self, bbox):
        return

    def prepare_input(self, queue):
        # queue , bboxes, complexity
        res = []
        for bbox in queue:
            overhead = bbox.extra_fields['overheads']
            bbox = bbox.bbox
            bbox[:, 0] = bbox[:, 0] / self.config.frame_height
            bbox[:, 1] = bbox[:, 1] / self.config.frame_width
            bbox[:, 2] = bbox[:, 2] / self.config.frame_height
            bbox[:, 3] = bbox[:, 3] / self.config.frame_width
            overhead = overhead / self.config.max_complexity
            bbox = torch.cat([bbox, overhead], dim=1)
            pad = torch.nn.ConstantPad2d((0, 0, 0, self.config.padding-bbox.shape[0]), 0.)
            bbox = pad(bbox)
            res.append(bbox)
        res = torch.stack(res)
        return res.view(1, self.config.max_queue_size, -1)

    def post_proc(self, output):
        output = output.squeeze(0)
        bbox = output[:, :4]
        overhead = output[:, -1]

        # clean padding bbox
        index = 0
        while index < bbox.shape[0]:
            if bbox[index][0] == bbox[index][1] == bbox[index][2] == bbox[index][3]:
                break
            index += 1
        return bbox[:index], overhead[:index]
