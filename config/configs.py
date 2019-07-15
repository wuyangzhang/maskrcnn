class Config:
    def __init__(self):

        # general
        self.frame_height = 375
        self.frame_width = 1242
        self.max_complexity = 50

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

        # flow control manager
        self.refresh_interval = 10