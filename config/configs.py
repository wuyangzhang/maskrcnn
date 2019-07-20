class Config:
    def __init__(self):
        # general
        self.frame_height = None
        self.frame_width = None
        self.max_complexity = 50
        self.use_local = True

        # dataset
        self.eval_dataset = 'kitti'
        self.kitti_video_path = '/home/wuyang/kitty/testing/seq_list.txt'

        # prediction manager
        self.model_path = '/home/wuyang/maskrcnn-benchmark/prediction/rppn_checkpoint.pth'
        self.pred_algos = ('lstm', 'convlstm')
        self.pred_algo = 0
        self.max_queue_size = 5
        self.padding = 32

        # partition manager
        self.par_num = 2
        self.total_remote_servers = 2
        self.servers = {0: ('127.0.0.1', 5050), 1: ('127.0.0.1', 5051)}

        # self.servers = {1: ('127.0.0.1', 5050), 2: ('127.0.0.1', 5051),
        #                 3: ('127.0.0.1', 5052), 4: ('127.0.0.1', 5052)}

        self.overlap_threshold = 0.7

        # flow control manager
        self.refresh_interval = 10
