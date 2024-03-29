import os
import cv2
import json
import time

import numpy as np

from pdf2image import convert_from_path
from difflib import SequenceMatcher


EMAIL_RGX = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'


def pdf_to_jpg(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    if not os.path.exists(file_path):
        print('Error: {}: file not found.'.format(file_path))
        return None

    pages = convert_from_path(file_path)

    pages_path = []
    for i, page in enumerate(pages):
        pages_path.append(os.path.join(dir_path, 'page_{}.jpg'.format(i)))
        page.save(pages_path[-1], 'JPEG')
    return pages_path


def save_cv_image(img, original_path, extension, del_original=False):
        new_path = original_path.split('.')[:-1]
        new_path[-1] += '_processed'
        new_path.append(extension)
        new_path = '.'.join(new_path)

        if os.path.exists(new_path):
            os.remove(new_path)
        if del_original:
            os.remove(original_path)
        cv2.imwrite(new_path, img)

        return new_path
    

def process_text(text_data):
    word_list = []
    conf_list = []
    bb_list = []
    parse_text = []
    parse_conf = []
    parse_bb = []
    last_word = ''
    for i, word in enumerate(text_data['text']):
        if word != '':
            word_list.append(word)
            conf_list.append(text_data['conf'][i])
            bb_list.append([text_data['left'][i],
                            text_data['top'][i],
                            text_data['width'][i],
                            text_data['height'][i]])
            last_word = word
        if (last_word != '' and word == '') or (i == len(text_data['text']) - 1):
            if len(word_list) > 0:
                parse_text.append(word_list)
                parse_conf.append(conf_list)
                parse_bb.append(bb_list)
            word_list = []
            conf_list = []
            bb_list = []
    return parse_text, parse_conf, parse_bb


def flatten(list_of_lists):
    """Flatten ND list to 1D list.

    Args:
        list_of_lists (list): List to flatten.

    Returns:
        list: Flattened 1D list.
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def get_json_from_file(filename):
    with open(filename) as json_file:
        return json.load(json_file)


def remove_background(img, kernel=(2, 1), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    border = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    dilation = cv2.dilate(resizing, kernel, iterations=iterations)
    erosion = cv2.erode(dilation, kernel, iterations=iterations)
    #* Image cleaning
    # Close small dots
    clean_dots = cv2.morphologyEx(src=erosion, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
    # Resharpen our text by making binary img
    cleaned = cv2.threshold(clean_dots, 170, 255, cv2.THRESH_BINARY)[1]
    return cleaned


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()