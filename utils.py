import os
import subprocess

import cv2 as cv


def remove_dot_background(img, kernel=(5, 5)):
    # Global thresholding and invert color
    ret1, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    # Removing noise
    cv_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)
    opening = cv.morphologyEx(th1, cv.MORPH_OPEN, cv_kernel)

    # Re-invert color
    res = 255 - opening

    return res


def pdf_to_tiff(file_path):
    if not os.path.exists(file_path):
        print('Error: {}: file not found.'.format(file_path))
        return None

    file_path_tiff = file_path.split('.')[:-1]
    file_path_tiff.append('tiff')
    file_path_tiff = '.'.join(file_path_tiff)

    if os.path.exists(file_path_tiff):
        os.remove(file_path_tiff)

    subprocess.run(['convert',
                    '-density', '600',
                    file_path,
                    '-background', 'white',
                    '-alpha', 'background',
                    '-alpha', 'off',
                    '-depth', '8',
                    file_path_tiff])
    
    return file_path_tiff

def save_cv_image(img, original_path, extension, del_original=False):
        new_path = original_path.split('.')[:-1]
        new_path.append(extension)
        new_path = '.'.join(new_path)

        if os.path.exists(new_path):
            os.remove(new_path)

        cv.imwrite(new_path, img)
        return new_path