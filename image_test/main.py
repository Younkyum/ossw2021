import cv2
from matplotlib import pyplot as plt

img_bgr = cv2.imread('S__26132483.jpg')

cv2.imshow("img_bgr", img_bgr)
# pause
keycode = cv2.waitKey(0)
# ESC key to exit
if keycode == 27:
    cv2.destroyAllWindows()

img_bitwise_not_bgr = cv2.bitwise_not(img_bgr)

cv2.imshow("img_bitwise_not_bgr", img_bitwise_not_bgr)
# pause
keycode = cv2.waitKey(0)
# ESC key to exit
if keycode == 27:
    cv2.destroyAllWindows()

img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)

cv2.imshow("img_bitwise_not_bgr2gray", img_bitwise_not_bgr2gray)
# pause
keycode = cv2.waitKey(0)
# ESC key to exit
if keycode == 27:
    cv2.destroyAllWindows()

ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 255, cv2.THRESH_BINARY)

cv2.imshow("img_binary", img_binary)
# pause
keycode = cv2.waitKey(0)
# ESC key to exit
if keycode == 27:
    cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contour = cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)

cv2.imshow("img_contour", img_contour)
# pause
keycode = cv2.waitKey(0)
# ESC key to exit
if keycode == 27:
    cv2.destroyAllWindows()
