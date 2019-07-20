import time
import cv2

from frame_filter import Filter
from state_monitor import ControlManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config
from demo import ApplicationManager
from dataset import MobiDistDataset


def main():

    mask_engine = ApplicationManager()

    for i in range(6):
        #file = '/home/wuyang/kitty/testing/image_02/0000/00002{}.png'.format(str(i+2))
        file = '/home/wuyang/kitty/testing/image_02/0002/00010{}.png'.format(str(i+2))

        img = cv2.imread(file)
        res, _, _ = mask_engine.run(img)
        #cv2.imshow("COCO detections", res)
        cv2.imwrite('/home/wuyang/' + str(i+10) + '.png', res)
        #cv2.waitKey()
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
