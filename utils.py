import os
import re
import json
import PyPDF2
import datetime
import subprocess

import cv2 as cv

EMAIL_RGX = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

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
                    '-density', '800',
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


def save_bb_image(img_path, bb):
    img = cv.imread(img_path)
    for row in bb:
        for box in row:
            (x, y, w, h) = (box[0], box[1], box[2], box[3])
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
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


def overlapping_lines_filter(lines, sort_idx):
    filtered = []
    
    lines = sorted(lines, key=lambda lines: lines[sort_idx])
    threshold = 10
    
    for i in range(len(lines)):
        l = lines[i]
        if i > 0:
            l_prev = lines[i-1]
            if l[sort_idx] - l_prev[sort_idx] > threshold:
                filtered.append(l)
        else:
            filtered.append(l)
    return filtered