import cv2 as cv
import numpy as np

def calc_angle(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    inv = cv.bitwise_not(gray)
    thresh = cv.threshold(inv, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_contour = contours[0]
    
    rect = cv.minAreaRect(largest_contour)

    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    return angle


def rotate_img(image, angle):
    new_image = image.copy()

    (h, w) = new_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1)

    new_image = cv.warpAffine(new_image, matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return new_image


if __name__ == '__main__':
    path = '/home/adrien/Repositories/Cezam/ocr-cezam/OCR/Relevés bancaires/CIC_C01_Relevé_Aout_2018/0000.jpg'

    image = cv.imread(path)

    print(image.shape)

    res, angle = calc_angle(image)

    res1 = rotate_img(image, angle)

    cv.imwrite('test.jpg', res1)
