import argparse
import time
from datetime import timedelta
import datetime

import torch
import torch.nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers.convlstm import ConvLSTM
from maskrcnn_benchmark.data.datasets.kitti.kitti_data import get_loader

# data load..

# prepare the training dataset..

# input & output

# input: K last frames. output : next frames

# what will the cost function for the training purpose? In other words, how to measure the similarity of two frames?

# what will be the training dataset??


class VideoNet(torch.nn.Module):

    def __init__(self, cfg):
        super(VideoNet, self).__init__()
        height, width, channels = 224, 224, 3

        self.convlstm = ConvLSTM(input_size=(height, width),
                 input_dim=channels,
                 hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

        self.conv1 = torch.nn.Conv3d(in_channels=cfg.PREDICTION.SEQ_LEN, out_channels=1, kernel_size=3, stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # need to reduce the high dimension...
        #  torch.Size([2, 5, 128, 224, 224])
        x = self.convlstm(x)[0][0]
        x = self.conv1(x).squeeze(1)
        x = self.conv2(x)
        return x

def train(cfg):

    # Load data
    train_loader = get_loader(seq_length=cfg.PREDICTION.SEQ_LEN, batch_size=cfg.PREDICTION.BATCH)

    # Build the model
    model = VideoNet(cfg)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    lr = cfg.PREDICTION.LEARNINGRATE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.cuda()
    loss_fn = loss_fn.cuda()


    for epoch in range(0, cfg.PREDICTION.EPOCHS):
        epoch_start_time = time.time()
        adjust_learning_rate(optimizer, lr, epoch)
        total_loss = _train(train_loader, model, loss_fn, optimizer, lr)
        elapsed_time = str(timedelta(seconds=int(time.time() - epoch_start_time)))

        print('epoch {} time: {} | mMSE {:.2e}'.format(epoch, elapsed_time, total_loss['mse']))
        if epoch % cfg.PREDICTION.SAVE_ITER == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'state_dict': model.state_dict(),
            })


def _train(train_loader, model, loss_fn, optimizer, lr):

    total_loss = {'mse': 0}

    for it, (input, label) in enumerate(train_loader):
        loss = 0
        input = input.cuda(async=True)
        label = label.cuda(async=True)
        #model.train()

        output = model(input)#.cuda())
        mse_loss = loss_fn(output, label)

        total_loss['mse'] += mse_loss.data

        loss += mse_loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if it % cfg.PREDICTION.SAVE_ITER == 0:
            total_loss['mse'] /= cfg.PREDICTION.SAVE_ITER
            print('Time {} : batch count {} | lr {:.3f} | mse {}'
                  .format(datetime.datetime.now(), it, lr, total_loss['mse']))
            for k in total_loss: total_loss[k] = 0
    return total_loss


def compute_loss(output, label):
    pass

def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    train(cfg)


if __name__ == "__main__":
    main()
