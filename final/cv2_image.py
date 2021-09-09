import cv2

sum = 0
total_pixel = 0
ave_c = 0

img = cv2.imread('test2.jpg', 0)

print(img)
for i in img:
    for j in i:
        total_pixel = total_pixel + 1
        sum = sum + j

ave_c = sum/total_pixel

for i in img:
    for j in i:
        if j > ave_c:
            print(1, end = ' ')
        else:
            print(0, end = ' ')
    print()
