import cv2


def calc_angle(img):
    inv = cv2.bitwise_not(img)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    rect = cv2.minAreaRect(largest_contour)

    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    # * Les images ne devraient pas être décalées de plus de 45°, donc on ne fait rien.
    if angle > 45 or angle < -45:
        angle = 0

    return angle


def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)

    img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img


def deskew_img(img):
    angle = calc_angle(img)
    
    # On ne rotationne pas l'image si l'angle est trop faible pour gagner du temps.
    if -0.1 < angle < 0.1:
        return img
    return rotate_img(img, angle)