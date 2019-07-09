import time
import cv2
from partition.partition_manager import PartitionManager
from demo.mobile_client import MaskCompute


file = 'COCO_test2014_000000032245.jpg' #baseball player
#file = 'COCO_test2014_000000005771.jpg' #clock

src = img = cv2.imread('/home/wuyang/coco/test2014/'+file)
img = cv2.imread('/home/wuyang/Downloads/testing/image_02/0014/000010.png')

engine = MaskCompute()

bbox, bbox_complexity = engine.run(img)
composite = engine.mask_overlay(img, bbox)
cv2.imwrite('/home/wuyang/partition_origin.png', composite)


# partition_num = 8
#
# partition_mgr = PartitionManager(parition_num=partition_num)
#
# imgs = partition_mgr.frame_partition(img, bbox.bbox, bbox_complexity, [0.1] * partition_num)
#
# res = []
# for i, img in enumerate(imgs):
#     # cv2.imshow("COCO detections", i)
#     # cv2.waitKey(1000000)
#     cv2.imwrite('/home/wuyang/partition{}.png'.format(i), img)
#     bbox, complexity = engine.run(img, resize=False)
#     res.append(bbox)
#     composite = engine.mask_overlay(img, bbox)
#     cv2.imwrite('/home/wuyang/partition{}_{}.png'.format(i, i), composite)
#
# mask = partition_mgr.merge_partition(res)
# res = engine.mask_overlay(src, mask, dist=True)
# cv2.imwrite('/home/wuyang/partitionxxx.png', res)


