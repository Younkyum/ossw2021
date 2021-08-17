import cv2
import numpy as np

img_rgb = cv2.imread('test1.png')
template = cv2.imread('/gdrive/MyDrive/cropStudy/answer_temp.png')
h, w = template.shape[:-1]

res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
threshold = .65
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):  # Switch collumns and rows
    print(pt)
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2_imshow(img_rgb)