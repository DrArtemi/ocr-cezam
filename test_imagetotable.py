import cv2
from PIL import Image
import io
import numpy as np
try:
    import urllib.request as urllib
except ModuleNotFoundError:
    import urllib

# read an image by filepath or image_url, im=filepath/image_url
def imgread(im):
    try:
        image = Image.open(io.BytesIO(urllib.urlopen(im).read()))
    except ValueError:
        try:
            image = Image.open(im)
        except FileExistsError:
            return None
    try:
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except:
        return None
    return image

# your image filepath or url
img = "/media/adrien/Shared/Work/Cezam/documents/OCR/Relevés bancaires/CIC_C01_Relevé_Juillet_2018/0001.jpg"

im = imgread(img)

# cv2.imshow('image', im)
# cv2.waitKey(0)

im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# cv2.imshow("gray image", im)
# cv2.waitKey(0)

dst = cv2.adaptiveThreshold(~im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

# cv2.imshow("binary image", dst)
# cv2.waitKey(0)

# copy dst, then for horizontal and vertical lines' detection.
horizontal = dst.copy()
vertical = dst.copy()
scale = 15  # play with this variable in order to increase/decrease the amount of lines to be detected

# Specify size on horizontal axis
print(horizontal.shape)
horizontalsize = horizontal.shape[1] // scale
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
cv2.imwrite("horizontal.jpg", horizontal)

# vertical
verticalsize = vertical.shape[0] // scale
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
cv2.imwrite("vertical.jpg", vertical)

# table line
table = horizontal + vertical
cv2.imwrite("table.jpg", table)

# the joint points between horizontal line and vertical line.
joints = cv2.bitwise_and(horizontal, vertical)
cv2.imwrite("joints.jpg", joints)
