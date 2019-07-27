from PIL import Image
import torch
from prediction import RPPNDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList


def visualize():
    test_video_files = '/home/wuyang/kitty/testing/seq_list.txt'
    dataset = 'kitti'
    eval_data_loader = RPPNDataset(test_video_files, dataset).getDataLoader(batch_size=1, shuffle=True)

    for batch_id, data in enumerate(eval_data_loader):
        train_x, train_y, path = data
        train_x = train_x.cuda()
        labels = train_y.cuda()
        path = [p[0].split('.')[0] + '.png' for p in path]
        imgs = [cv2.imread(p) for p in path]

        # historical bbox
        train_x = torch.squeeze(train_x)
        train_x = train_x.reshape(train_x.shape[0], -1, 5)[:, :, :4]
        train_x[:, :, 0] *= imgs[0].shape[1]
        train_x[:, :, 1] *= imgs[0].shape[0]
        train_x[:, :, 2] *= imgs[0].shape[1]
        train_x[:, :, 3] *= imgs[0].shape[0]
        hist_bbox = [BoxList(box, imgs[0].shape[:2]) for box in train_x]

        hist_image = [render_bbox(hist_bbox[i], imgs[i]) for i in range(len(hist_bbox))]
        _ = [plt.imshow(image) for image in hist_image]

        bbox_label = train_y[0]




    # plt.imshow(image)
    # plt.show()


def render_bbox(bbox, image):
    boxes = bbox.bbox
    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple((252, 7, 3)), 1
        )

    return image

visualize()