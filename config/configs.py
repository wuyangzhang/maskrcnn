from pathlib import Path


class Config:
    #score_threshold = 0.5

    def __init__(self):
        # general
        self.frame_height = None
        self.frame_width = None
        self.max_complexity = 50
        self.use_local = True

        self.home_addr = str(Path.home())

        # self.home_addr = '/home/user/wz1_willRemoveOnJuly31/'
        # dataset
        self.datasets = ['kitti', 'davis']
        self.eval_dataset = 'davis'
        # kitti, h: 375, w: 1242
        self.kitti_video_path = self.home_addr + '/kitty/testing/seq_list.txt'
        self.davis_video_path = self.home_addr + '/datasets/davis/DAVIS/JPEGImages/480p/'

        # prediction manager
        self.model_path = self.home_addr + '/maskrcnn-benchmark/prediction/models/' \
                                           'lstm_single_checkpoint15.pth'

        self.pred_algos = ('lstm', 'convlstm')
        self.pred_algo = 0
        self.window_size = 4
        self.padding = 32

        # partition manager
        self.par_num = 2
        self.total_remote_servers = 2
        #self.servers = {0: ('192.168.55.100', 5053), 1: ('192.168.55.100', 5054)}
        self.servers = {0: ('192.168.55.100', 5051), 1: ('192.168.55.100', 5055)}

        self.overlap_threshold = 0.7

        # flow control manager
        self.refresh_interval = 10

        self.log = None