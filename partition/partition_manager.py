import random
import collections

import torch
import torch.nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


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

    @staticmethod
    def proc_capability_eval(proc_capability):
        '''
            Evaluate the processing capability based on the e2e latency
        '''
        max_latency = max(proc_capability)
        return [(1e-3 + max_latency) / latency for latency in proc_capability]

    @staticmethod
    def bbox_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def frame_crop(frame, bbox):
        return frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    def partition_to_nodes(self, node_num):
        '''mapping a partition to a specific node

            partition i belongs the node mapping[i]
        '''
        if self.server_par_map is None:
            order = [_ for _ in range(node_num)]
            random.shuffle(order)
            self.server_par_map = {order[i]: i for i in range(node_num)}
            self.par_server_map = {i: order[i] for i in range(node_num)}

    def frame_partition(self, frame, bbox, complexity_weights, proc_capability):
        """A frame partition scheme.

        This frame partition scheme performs based on the position of bounding boxes(bbox),
        the weights of bbox that indicate the potential computing complexity, and
        the available computing resources that are represented by the historical
        computing time. Combining all of them together, we intend to partition a frame
        by following two rules: (1) to minimize the impact on the performance(accuray),
        (2) to balance the workload of multiple partitions.

        step 1. Equal partition.

        step 2. Computation complexity aware placement.
        for each bbox, check whether it is overlapped with multiple partitions.
        if not, add it to that partition and change the partition weight.
        if yes, select one of partitions based on its current weight. Each partition should
        have equal probability to be selected.

        :param frame:      The frame we need to partition.
        :param bbox: The number of output channels for the convolution.
        :param weights: Spatial size of the convolution kernel.
        :param resources:       Additional position arguments forwarded to slim.conv2d.

        :return N partitions
        """
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

    def merge_partition(self, distributed_res):
        '''Merge results from distribution

            Returns bbox with the offset compensation & merged mask
            assume the distributed_res stores the results in the server order
        '''
        bboxes = []
        extras = collections.defaultdict(list)
        for server_id in distributed_res:
            bbox = distributed_res[server_id][0]
            # print('server processing', bbox.bbox)
            if len(bbox.bbox) == 0:
                continue
            # modify the bounding box positions by compensating the offsets
            par_id = self.server_par_map[server_id]
            x, y = self.partition_offset[par_id]
            bbox.bbox = bbox.bbox.int()
            bbox.add_offset(x, y)

            bboxes.append(bbox.bbox)

            # mask fixing.. put 0 around mask
            # find the offset for each mask in the order of (l, r, u, d)
            pad = torch.nn.ConstantPad2d((y,
                                          self.config.frame_width - y - bbox.extra_fields['mask'].shape[3],
                                          x,
                                          self.config.frame_height - x - bbox.extra_fields['mask'].shape[2]),
                                         0)

            bbox.extra_fields['mask'] = pad(bbox.extra_fields['mask'][:, :, ])

            # add extra fields
            for key in bbox.extra_fields.keys():
                extras[key].append(bbox.extra_fields[key])

        bboxes = torch.cat(bboxes, dim=0).float()

        # concatenate
        for key in extras.keys():
            extras[key] = torch.cat(extras[key], dim=0)

        # labels = torch.cat(extras['labels'], dim=0)
        # scores = torch.cat(extras['scores'], dim=0)
        # mask = torch.cat(extras['mask'], dim=0)
        # overheads = torch.cat(extras['overheads'], dim=0)

        # nms
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = torch.argsort(y2)

        pick = torch.zeros(idxs.shape).byte()
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick[i] = 1

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = torch.max(x1[i], x1[idxs[:]])
            yy1 = torch.max(y1[i], y1[idxs[:]])
            xx2 = torch.min(x2[i], x2[idxs[:]])
            yy2 = torch.min(y2[i], y2[idxs[:]])

            # compute the width and height of the bounding box
            w = torch.max(xx2 - xx1 + 1, torch.tensor([0.]))
            h = torch.max(yy2 - yy1 + 1, torch.tensor([0.]))

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:]]

            # delete all indexes from the index list that have
            idxs = idxs[overlap < self.config.overlap_threshold]

        # now 'pick' keeps the index of bbox that should remain
        # so we only keep those bbox and also update their extra fields.

        bbox = BoxList(bboxes[pick], extras['mask'].shape[2:][::-1])

        for key in extras.keys():
            bbox.extra_fields[key] = extras[key][pick]
        # bbox.extra_fields['labels'] = labels[pick]
        # bbox.extra_fields['scores'] = scores[pick]
        # bbox.extra_fields['mask'] = mask[pick]
        # bbox.extra_fields['overheads'] = overheads[pick]
        print('server processing', bbox.bbox)
        return bbox
