import torch.nn.utils.rnn as rnn



def pack_padded_seq(input):
    input[:, :, :, 0] + input[:, :, :, 1] + input[:, :, :, 2]  + input[:, :, :, 3]  != 0
    rnn.pack_padded_sequence()