def make_calimg(img):
    calimg = []
    imgline = []
    sum = 0
    total_pixel = 0
    for i in img:
        for j in i:
            total_pixel = total_pixel + 1
            sum = sum + j
    ave_c = sum / total_pixel

    for i in img:
        for j in i:
            if j > ave_c:
                imgline.append(1)
            else:
                imgline.append(0)
        calimg.append(imgline)
        imgline = []
    return calimg

def show_calimg(img):
    sum = 0
    total_pixel = 0
    for i in img:
        for j in i:
            total_pixel = total_pixel + 1
            sum = sum + j

    ave_c = sum / total_pixel

    for i in img:
        for j in i:
            if j > ave_c:
                print(1, end=' ')
            else:
                print(0, end=' ')
        print()