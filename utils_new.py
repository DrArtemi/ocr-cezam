import os
import re
import json
import math
import time
import datetime
import subprocess

import cv2 as cv
import numpy as np

from skimage import morphology
from pdf2image import convert_from_path

EMAIL_RGX = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def remove_dot_background(img, kernel=(5, 5)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    inv = cv.bitwise_not(gray)
    # Global thresholding and invert color
    ret1, th1 = cv.threshold(inv, 250, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Removing noise
    cv_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)
    opening = cv.morphologyEx(th1, cv.MORPH_OPEN, cv_kernel, iterations=1)
    
    cleaned = morphology.remove_small_objects(opening > 100, min_size=100, connectivity=1)
    cleaned = cleaned.astype('uint8') * 255
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, cv_kernel, iterations=1)

    
    # Re-invert color
    res = 255 - cleaned

    return res


def calc_angle(img):
    inv = cv.bitwise_not(img)
    thresh = cv.threshold(inv, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_contour = contours[0]
    
    rect = cv.minAreaRect(largest_contour)

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
    matrix = cv.getRotationMatrix2D(center, angle, 1)

    img = cv.warpAffine(img, matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return img


def deskew_img(img):
    angle = calc_angle(img)
    
    # On ne rotationne pas l'image si l'angle est trop faible pour gagner du temps.
    if -0.1 < angle < 0.1:
        return img
    return rotate_img(img, angle)


def pdf_to_jpg(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    if not os.path.exists(file_path):
        print('Error: {}: file not found.'.format(file_path))
        return None

    pages = convert_from_path(file_path)

    pages_path = []
    for i, page in enumerate(pages):
        pages_path.append(os.path.join(dir_path, 'page_{}.jpg').format(i))
        page.save(pages_path[-1], 'JPEG')
    return pages_path


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
                    '-density', '300',
                    file_path,
                    '-background', 'white',
                    '-alpha', 'background',
                    '-alpha', 'off',
                    file_path_tiff])
    
    return file_path_tiff


def save_cv_image(img, original_path, extension, del_original=False):
        new_path = original_path.split('.')[:-1]
        new_path.append(extension)
        new_path = '.'.join(new_path)

        if os.path.exists(new_path):
            os.remove(new_path)

        cv.imwrite(new_path, img)

        if del_original:
            os.remove(original_path)
        return new_path


def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):
    paths = []
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(input_pdf_path, "rb") as input_stream:
        input_pdf = PyPDF2.PdfFileReader(input_stream)

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        # ! : Il faut trouver une solution pour les documents pdf encryptés
        # if input_pdf.isEncrypted:
        #     print('Coucou')
        #     input_pdf.decrypt('')
        # print(input_pdf.getNumPages(), input_pdf.flattenedPages, input_pdf.isEncrypted)
        # exit()
        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            paths.append(file_name)
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)
    
    return paths


def write_file_from_text(text, filename):
    with open(filename, "w+") as file:
        file.write(text)


def save_bb_image_old(img_path, data, conf_thresh=50):
    img = cv.imread(img_path)
    for i, text in enumerate(data['text']):
        if int(data['conf'][i]) > conf_thresh:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    bb_img_path = img_path.split('.')
    bb_img_path[-2] += '_bb'
    bb_img_path = '.'.join(bb_img_path)
    if os.path.exists(bb_img_path):
        os.remove(bb_img_path)
    cv.imwrite(bb_img_path, img)


def save_bb_image(img_path, bb, v_lines, h_lines):
    img = cv.imread(img_path)
    for row in bb:
        for box in row:
            (x, y, w, h) = (box[0], box[1], box[2], box[3])
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for line in v_lines:
        img = cv.line(img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
    for line in h_lines:
        img = cv.line(img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
    bb_img_path = img_path.split('.')
    bb_img_path[-2] += '_bb'
    bb_img_path = '.'.join(bb_img_path)
    if os.path.exists(bb_img_path):
        os.remove(bb_img_path)
    cv.imwrite(bb_img_path, img)


def get_json_from_file(filename):
    with open(filename) as json_file:
        return json.load(json_file)


def get_bank_id(text):
    data = get_json_from_file('bank_configs/banks.json')
    for line in text:
        for b_id, b_list in data.items():
            for b_pat in b_list:
                if b_pat in ' '.join(line).lower():
                    return b_id
    return None


def get_client_name(text):
    for line in text:
        for i, w in enumerate(line):
            if 'mme' == w.lower():
                return line[i + 1], line[i + 2]
    return None, None


def get_words_in_zone(w_list, bb, zone):
    words = []
    for i in range(len(w_list)):
        if zone[0] < bb[i][0] < zone[1]:
            words.append(w_list[i])
    return words


def get_addresses(text, bb, size, bank_utils, dicts):
    bank_address = None
    client_address = None
    bi_bank = bank_utils['bank_info']
    bi_client = bank_utils['client_info']
    for i in range(len(text)):
        for a_type in dicts['address']:
            for j, w in enumerate(text[i]):
                if a_type == w.lower():
                    if (bi_bank[0][0] * size[0]) < bb[i][j][0] < (bi_bank[0][1] * size[0]) and \
                        (bi_bank[1][0] * size[1]) < bb[i][j][1] < (bi_bank[1][1] * size[1]):
                        bank_address = ' '.join(
                            get_words_in_zone(text[i], bb[i], (bi_bank[0][0] * size[0], bi_bank[0][1] * size[0])) +
                            get_words_in_zone(text[i+1], bb[i+1], (bi_bank[0][0] * size[0], bi_bank[0][1] * size[0]))
                        )
                    elif (bi_client[0][0] * size[0]) < bb[i][j][0] < (bi_client[0][1] * size[0]) and \
                        (bi_client[1][0] * size[1]) < bb[i][j][1] < (bi_client[1][1] * size[1]):
                        client_address = ' '.join(
                            get_words_in_zone(text[i], bb[i], (bi_client[0][0] * size[0], bi_client[0][1] * size[0])) +
                            get_words_in_zone(text[i+1], bb[i+1], (bi_client[0][0] * size[0], bi_client[0][1] * size[0]))
                        )
    return bank_address, client_address


def get_agency_phone(text, bb, size, bank_utils, dicts):
    bi = bank_utils['bank_info']
    tel = None
    for i in range(len(text)):
        for p_type in dicts['phone']:
            for j, w in enumerate(text[i]):
                if p_type == w.lower():
                    if (bi[0][0] * size[0]) < bb[i][j][0] < (bi[0][1] * size[0]) and \
                        (bi[1][0] * size[1]) < bb[i][j][1] < (bi[1][1] * size[1]):
                        tel = get_words_in_zone(text[i], bb[i], (bi[0][0] * size[0], bi[0][1] * size[0]))
                        tel = ''.join(tel[1:]).replace(':', '')
    return tel


def get_agency_email(text, bb, size, bank_utils):
    bi = bank_utils['bank_info']
    email = None
    for i in range(len(text)):
        for j, w in enumerate(text[i]):
            if re.search(EMAIL_RGX, w):
                if (bi[0][0] * size[0]) < bb[i][j][0] < (bi[0][1] * size[0]) and \
                    (bi[1][0] * size[1]) < bb[i][j][1] < (bi[1][1] * size[1]):
                    email = w
    return email


def get_date(text, bb, size, bank_utils):
    dp = bank_utils['date_format']
    bi = bank_utils['date_info']
    date = None
    for i in range(len(text)):
        for j, w in enumerate(text[i]):
            try:
                tmp_date = datetime.datetime.strptime(w, dp)
                if (bi[0][0] * size[0]) < bb[i][j][0] < (bi[0][1] * size[0]) and \
                    (bi[1][0] * size[1]) < bb[i][j][1] < (bi[1][1] * size[1]):
                    date = tmp_date
            except ValueError:
                continue
    
    if date is not None:
        return date.strftime('%d'), date.strftime('%B'), date.strftime('%Y')
    return None, None, None


def get_table_columns(text, bb, shape, v_lines):
    line_idx = None
    for i in range(len(text)):
        for j, w in enumerate(text[i]):
            if shape[2] < bb[i][j][0] < shape[3] and\
                 shape[0] < bb[i][j][1] < shape[1]:
                line_idx = i
                break
        if line_idx is not None:
            break
    if line_idx is None:
        return None
    columns = []
    for i in range(len(v_lines) - 1):
        col_title = [str(i)]
        for j, w in enumerate(text[line_idx]):
            if v_lines[i][0] < bb[line_idx][j][0] < v_lines[i+1][0]:
                col_title.append(w)
        columns.append(' '.join(col_title) if len(col_title) > 0 else 'Unknown')
        
    return columns


def get_rows(text, bb, shape):
    for i in range(len(text)):
        for j, w in enumerate(text[i]):
            if shape[2] < bb[i][j][0] < shape[3] and\
                 shape[0] < bb[i][j][1] < shape[1]:
                print(text[i])
                break


def is_line_vertical(line):
    return line[0] == line[2]


def is_line_horizontal(line):
    return line[1] == line[3]


def overlapping_lines_filter(lines, sort_idx, threshold=50):
    filtered = []
    other_idx = 0 if sort_idx == 1 else 1
    lines = sorted(lines, key=lambda lines: lines[sort_idx])
    
    for i in range(len(lines)):
        l = lines[i]
        if i > 0:
            if l[sort_idx] - filtered[-1][sort_idx] > threshold:
                filtered.append(l)
            else:
                # if abs(l[other_idx] - l[other_idx+2]) > abs(l_prev[other_idx] - l_prev[other_idx+2]):
                if sort_idx == 0:
                    if not filtered[-1][other_idx] < l[other_idx+2] or not l[other_idx] < filtered[-1][other_idx+2]:
                        filtered[-1][other_idx] = max(l[other_idx], filtered[-1][other_idx])
                        filtered[-1][other_idx+2] = min(l[other_idx+2], filtered[-1][other_idx+2])
                else:
                    if not filtered[-1][other_idx+2] < l[other_idx] or not l[other_idx+2] < filtered[-1][other_idx]:
                        filtered[-1][other_idx] = min(l[other_idx], filtered[-1][other_idx])
                        filtered[-1][other_idx+2] = max(l[other_idx+2], filtered[-1][other_idx+2])
        else:
            filtered.append(l)
    return filtered


def pt_line_collision(line, point, threshold=50):
    dist_1 = distance(point, (line[0], line[1]))
    dist_2 = distance(point, (line[2], line[3]))

    line_len = distance((line[0], line[1]), (line[2], line[3]))

    if dist_1 + dist_2 >= line_len - threshold and\
         dist_1 + dist_2 <= line_len + threshold:
        return True
    return False


def irrelevant_lines_filter(v_lines, h_lines, threshold=20):
    filtered_v_lines = []
    filtered_h_lines = []
    for v_line in v_lines:
        for h_line in h_lines:
            if pt_line_collision(h_line, (v_line[0], v_line[1])) or\
                pt_line_collision(h_line, (v_line[2], v_line[3])):
                filtered_v_lines.append(v_line)
                break
    for h_line in h_lines:
        for v_line in filtered_v_lines:
            if pt_line_collision(v_line, (h_line[0], h_line[1])) or\
                pt_line_collision(v_line, (h_line[2], h_line[3])):
                filtered_h_lines.append(h_line)
                break
    return filtered_v_lines, filtered_h_lines
