from maskrcnn_benchmark.modeling.prediction.frame_predict import WorkloadDistPrediction


class PredictionManager:

    def __init__(self, cfg):

        self.max_queue_size = 5

        self.mask_queue = list()
        self.predict_engine = WorkloadDistPrediction(cfg)
        self.next_predict = None

    def add_mask(self, bbox, unit):
        if len(self.mask_queue) == self.max_queue_size:
            self.mask_queue.pop(0)
        self.mask_queue.append((bbox.bbox, unit))

        if len(self.mask_queue) == self.max_queue_size:
            #self.next_predict = self.predict_workload_dist()
            self.next_predict = self.test_predict()

    def get_queue_len(self):
        return len(self.mask_queue)

    def get_last_mask(self):
        return self.mask_queue[-1]

    '''
    Predict computation cost in pixel level
    
    resize => avg => bbox area avg
    
    Args:  
        @units: recording the number of layers that the backbone network actually runs upon each ResNet block.   

    Return:
        normalize the unit input apply that to each bbox
    '''
    def predict_workload_dist(self, bbox, unit):

        res = self.predict_engine.run(bbox, unit)
        count = 0
        for i in range(len(res)):
            bbox[i] = res[i][:4]
            unit[i] = res[i][-1]
            count += 1
        bbox = bbox[:count]
        unit = unit[:count]
        return bbox, unit

    def test_predict(self):
        return self.mask_queue[-1]

    def next_predict_workload_dist(self):
        return self.next_predict

    '''
    A pixel level computation cost normalizer.
    resize => avg => bbox area avg
    Args:  
        @units: recording the number of layers that the backbone network actually runs upon each ResNet block.   

    Return:
        normalize the unit input apply that to each bbox
    '''

    def unit_calculator(self, bbox, units):
        return units[0]