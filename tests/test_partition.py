from partition.partition_manager import PartitionManager
import cv2
import time

p = PartitionManager()
frame = cv2.imread('/Users/wuyang/Desktop/dog.jpg')

bbox = [[40, 40, 160, 160], [300, 300, 360, 360], [100, 100, 400, 400]]
weight = [3.5, 2.5, 1.5]
proc_capability = [10, 20, 30, 40]


for i in range(10):
    s = time.time()
    res = p.frame_partition(frame, bbox, weight, proc_capability)
    print(res, time.time() - s)

cv2.imshow('title', frame)
print('s')
cv2.waitKey()
cv2.destroyAllWindows()