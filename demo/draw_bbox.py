import matplotlib.pyplot as plt
import cv2

image = cv2.imread('/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/bus/00060.jpg')


top_left = [40, 40]
bottom_right = [240, 240]

image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255, 255, 255), 5
            )

plt.imshow(image)
plt.show()