import torch
from prediction import RPPNDataset
import matplotlib.pyplot as plt
import cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList
from prediction.lstm_single import LSTM
from config import Config
from prediction.preprocesing1 import reorder

torch.manual_seed(0)

config = Config()
alg = 1
if alg == 1:
    model = LSTM(input_size=4, hidden_size=16, window=config.window_size, num_layers=2).cuda()
    model.load_state_dict(torch.load(config.model_path))


def visualize():
    test_video_files = config.home_addr + 'kitty/testing/seq_list.txt'
    dataset = 'kitti'
    eval_data = RPPNDataset(test_video_files, dataset)
    eval_data_loader = eval_data.getDataLoader(batch_size=1, window_size=config.window_size, shuffle=True)

    for batch_id, data in enumerate(eval_data_loader):
        train_x, train_y, path = data
        train_x = train_x.cuda()
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 5)

        train_x = train_x[:, :, :, :4]
        _train_x = train_x.clone()

        train_y = train_y.cuda()
        path = [p[0].split('.')[0] + '.png' for p in path]
        imgs = [cv2.imread(p) for p in path]
        label_img = imgs[-1]
        #pred_img = label_img.copy()
        pred_img = imgs[-2].copy()

        train_x = train_x.squeeze()
        train_x[:, :, 0] *= imgs[0].shape[1]
        train_x[:, :, 1] *= imgs[0].shape[0]
        train_x[:, :, 2] *= imgs[0].shape[1]
        train_x[:, :, 3] *= imgs[0].shape[0]
        hist_bbox = [BoxList(box, imgs[0].shape[:2]) for box in train_x]

        hist_image = [render_bbox(hist_bbox[i], imgs[i]) for i in range(len(hist_bbox))]
        for image in hist_image:
            b, g, r = cv2.split(image)  # get b,g,r
            image = cv2.merge([r, g, b])  # switch it to rgb
            plt.imshow(image)
            plt.show()

        # show the ground truth result
        train_y = torch.squeeze(train_y)[:, :4]
        train_y[:, 0] *= imgs[0].shape[1]
        train_y[:, 1] *= imgs[0].shape[0]
        train_y[:, 2] *= imgs[0].shape[1]
        train_y[:, 3] *= imgs[0].shape[0]

        label_bbox = BoxList(train_y, imgs[0].shape[:2])
        label_image = render_bbox(label_bbox, label_img)
        b, g, r = cv2.split(label_image)  # get b,g,r
        label_image = cv2.merge([r, g, b])  # switch it to rgb
        plt.imshow(label_image)
        plt.show()

        # show the prediction result
        if alg == 1:
            x = reorder(_train_x)
            x = x.reshape(x.shape[0], x.shape[1], -1, 4)

            x = x.permute(0, 2, 1, 3)

            nonpad = x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2] + x[:, :, :, 3] != 0
            x = x[nonpad].reshape(-1, config.window_size, 4)

            if len(x) == 0:
                continue
            out = model(x)

        out[:, 0] *= imgs[0].shape[1]
        out[:, 1] *= imgs[0].shape[0]
        out[:, 2] *= imgs[0].shape[1]
        out[:, 3] *= imgs[0].shape[0]

        pred_bbox = BoxList(out, imgs[0].shape[:2])
        pred_image = render_bbox(pred_bbox, pred_img)
        b, g, r = cv2.split(pred_image)  # get b,g,r
        pred_image = cv2.merge([r, g, b])  # switch it to rgb
        plt.imshow(pred_image)
        plt.show()

        a = 1


def render_bbox(bbox, image):
    boxes = bbox.bbox
    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple((255, 0, 0)), 3
        )

    return image


visualize()
