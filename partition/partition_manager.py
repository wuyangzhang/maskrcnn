import random
import torch
import cv2
from maskrcnn_benchmark.utils import cv2_util


class PartitionManager:
    def __init__(self, config):
        self.config = config
        self.par_num = self.config.par_num
        assert self.par_num % 2 == 0, 'Error: partition number must be the time of 2'

        self.server_par_map = None
        # x * y = N, min abs(x-y)
        self.height_partition = self.width_partition = 1
        target = float('inf')
        for num in range(1, self.par_num):
            if self.par_num % num == 0:
                tmp = abs(self.par_num / num - num)
                if tmp < target:
                    target = tmp
                    self.height_partition = num
        self.width_partition = self.par_num // self.height_partition
        self.partition_offset = list()

    @staticmethod
    def find_intersection(a, b):  # returns None if rectangles don't intersect
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    ''' 
        Evaluate the processing capability based on the e2e latency 
    '''
    @staticmethod
    def proc_capability_eval(proc_capability):
        max_latency = max(proc_capability)
        return [(1e-3 + max_latency) / latency for latency in proc_capability]

    @staticmethod
    def bbox_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def frame_crop(frame, bbox):
        return frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    '''mapping a partition to a specific node
        
        partition i belongs the node mapping[i]
    '''

    def partition_to_nodes(self, node_num):
        if self.server_par_map is None:
            order = [_ for _ in range(node_num)]
            random.shuffle(order)
            self.server_par_map = {order[i]: i for i in range(node_num)}
            self.par_server_map = {i: order[i] for i in range(node_num)}

    """A frame partition scheme.  

    This frame partition scheme performs based on the position of bounding boxes(bbox),  
    the weights of bbox that indicate the potential computing complexity, and  
    the avaiable computing resources that are represented by the historical  
    computing time. Combining all of them toghter, we intend to partition a frame   
    by following two rules: (1) to minimize the impact on the performance(accuray),   
    (2) to balance the workload of multiple partitions.   

    step 1. Equal partition.   

    step 2. Computation complexity aware placement.  
    for each bbox, check whether it is overlapped with multiple partitions.   
    if not, add it to that partition and change the partition weight.   
    if yes, select one of partitions based on its current weight. Each partition should 
    have equal probability to be selected.  


    Args:  
         frame:      The frame we need to partition.  
         bbox: The number of output channels for the convolution.  
         weights: Spatial size of the convolution kernel.  
         resources:       Additional position arguments forwarded to slim.conv2d.  

      Returns:  
         N partitions  

    """

    def frame_partition(self, frame, bbox, complexity_weights, proc_capability):

        # each processing unit should randomly select a partition in the beginning.
        self.partition_to_nodes(len(proc_capability))
        mapping = self.par_server_map

        proc_capability = self.proc_capability_eval(proc_capability)

        # weight_correction
        alpha = sum(proc_capability) / sum(
            [self.bbox_area(bbox) * complexity_weights[i] for i, bbox in enumerate(bbox)]
        )

        # initialize the coordinates of partitions by equally segmenting them.
        partition_coordinates = []
        height, width, _ = frame.shape
        height_unit, width_unit = height // self.height_partition, width // self.width_partition
        for i in range(self.height_partition):
            for j in range(self.width_partition):
                partition_coordinates.append([i * height_unit, j * width_unit,
                                              (i + 1) * height_unit, (j + 1) * width_unit])

        for box in bbox:
            box = box.to(torch.int64)
            overlap = []  # put the id of the overlapped partition and the overlapped size
            for i, par in enumerate(partition_coordinates):
                intersect_size = self.find_intersection(box, par)
                # this box is fully covered by that partition.. just update its capability
                box_size = self.bbox_area(box)

                # no intersection bewteen the bbox and the partition area
                if intersect_size is None:
                    continue
                # the bbox must be fully covered by the partition area
                if intersect_size == box_size:
                    proc_capability[mapping[i]] -= alpha * intersect_size
                    break

                overlap.append((i, intersect_size))

                # decide which partition should handle this box and then change its size..
            # prefix sum of proc_capability
            if len(overlap) == 0:
                continue
            prefix = [proc_capability[mapping[overlap[0][0]]]] * len(overlap)
            for i in range(1, len(overlap)):
                prefix[i] = prefix[i - 1] + proc_capability[mapping[overlap[i][0]]]
            num = random.uniform(0, prefix[-1])
            l, r = 0, len(prefix)
            while l < r:
                m = l + (r - l) // 2
                if prefix[m] < num:
                    l = m + 1
                else:
                    r = m
            if l == len(prefix):
                l -= 1

            # overlap[l][0] will be the partition id to cover this box.
            par_id = overlap[l][0]
            # update the capability..
            proc_capability[mapping[par_id]] -= alpha * overlap[l][1]

            # resize the partition in order to fully cover the bbox..
            partition_coordinates[par_id][0] = min(partition_coordinates[par_id][0], box[0].data.tolist())
            partition_coordinates[par_id][1] = min(partition_coordinates[par_id][1], box[1].data.tolist())
            partition_coordinates[par_id][2] = max(partition_coordinates[par_id][2], box[2].data.tolist())
            partition_coordinates[par_id][3] = max(partition_coordinates[par_id][3], box[3].data.tolist())

        self.partition_offset = [(par[0], par[1]) for par in partition_coordinates]

        return [self.frame_crop(frame, partition_coordinates[self.server_par_map[i]]) for i in
                range(len(proc_capability))]

    '''Merge results from distribution
     
        Returns bbox with the offset compensation & merged mask
        assume the distributed_res stores the results in the server order  
    '''

    def merge_partition(self, distributed_res):
        res = None
        contours_list = list()
        for server_id in range(len(distributed_res)):
            par_id = self.server_par_map[server_id]
            x, y = self.partition_offset[par_id]

            # fix the bounding box positions by compensating the offsets
            for id, bbox in enumerate(distributed_res[server_id].bbox):
                bbox = bbox.to(torch.int64)
                distributed_res[server_id].bbox[id][0] = bbox[0] + y
                distributed_res[server_id].bbox[id][1] = bbox[1] + x
                distributed_res[server_id].bbox[id][2] = bbox[2] + y
                distributed_res[server_id].bbox[id][3] = bbox[3] + x

            if res is None:
                res = distributed_res[server_id]
            else:
                # merge bbox
                res.bbox = torch.cat((res.bbox, distributed_res[server_id].bbox))
                # merge labels
                res.extra_fields["labels"] = torch.cat(
                    (res.get_field("labels"), distributed_res[server_id].get_field("labels")))
                # merge scores
                res.extra_fields["scores"] = torch.cat(
                    (res.get_field("scores"), distributed_res[server_id].get_field("scores")))

            # merge mask
            if len(distributed_res[server_id].get_field('mask')) == 0:
                continue
            else:
                masks = distributed_res[server_id].extra_fields['mask'].numpy()
                for id, mask in enumerate(masks):
                    thresh = mask[0, :, :, None]
                    contours, hierarchy = cv2_util.findContours(
                        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for i, coord in enumerate(contours[0]):
                        contours[0][i][0][0] = coord[0][0] + y
                        contours[0][i][0][1] = coord[0][1] + x

                    contours_list.append(contours)
        res.extra_fields['mask'] = (contours_list)
        return res
