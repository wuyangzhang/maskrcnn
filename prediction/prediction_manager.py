from prediction.pred_delegator import PredictionDelegator


class PredictionManager:
    def __init__(self, config):

        self.config = config
        self.bbox_queue = list()
        self.max_queue_size = config.window_size
        self.pred_delegator = PredictionDelegator(config)
        self.img = None
        self.next_predict = None

    def add_bbox(self, bbox):

        # skip if no object has been found.
        if len(bbox.bbox) == 0:
            return

        # only keep the last max_queue_size results.
        if len(self.bbox_queue) == self.max_queue_size:
            self.bbox_queue.pop(0)
        self.bbox_queue.append(bbox)
        #print(self.bbox_queue)
            # self.next_predict = self.test_predict()

    def get_queue_len(self):
        return len(self.bbox_queue)

    def is_active(self):
        #print(self.bbox_queue, self.max_queue_size)
        return len(self.bbox_queue) >= self.max_queue_size

    def predict_bbox(self):
        '''
        Predict computation cost in pixel level

        resize => avg => bbox area avg

        :param units: recording the number of layers that
        the backbone network actually runs upon each ResNet block.

        :return:
            normalize the unit input apply that to each bbox
        '''
        return self.pred_delegator.run(self.bbox_queue)

    def test_predict(self):
        return self.bbox_queue[-1]

    def get_pred_bbox(self):
        '''
        Get the prediction of the next workload distribution.

        :return bbox coordinates
        :return bbox weights
        '''

        self.next_predict = self.predict_bbox()
        coords = self.next_predict
        weights = [1] * coords.shape[0]
        return coords, weights

    def unit_calculator(self, bbox, units):
        '''
        A pixel level computation cost normalizer.
        resize => avg => bbox area avg
        Args:
            @units: recording the number of layers that
            the backbone network actually runs upon each ResNet block.

        Return:
            normalize the unit input apply that to each bbox
        '''
        return units[0]
