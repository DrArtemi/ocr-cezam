import os
import cv2
import json
import time

import numpy as np

from pdf2image import convert_from_path


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


def remove_background(img, kernel=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    border = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    dilation = cv2.dilate(resizing, kernel,iterations=1)
    erosion = cv2.erode(dilation, kernel,iterations=1)
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


def valid_value(val, dates):
    if val == '':
        return False
    return len([d for d in dates if d != '' and 'solde' not in d.lower()]) > 0


def check_solde(tables, dicts):
    cred_regx = '|'.join(dicts['credit'])
    deb_regx = '|'.join(dicts['debit'])
    date_regx = 'date'
    solde_regx = 'solde'
    status = [None] * len(tables)
    for i, table in enumerate(tables):
        table.columns = table.columns.str.strip().str.lower()
        #* Get credit and debit column names
        cred_col = table.filter(regex=cred_regx)
        deb_col = table.filter(regex=deb_regx)
        date_col = table.filter(regex=date_regx)
        
        col_empty = [cred_col.empty, deb_col.empty, date_col.empty]
        if any(col_empty):
            col_names = ['credit', 'debit', 'date']
            missing_col = [col for j, col in enumerate(col_names) if col_empty[j]]
            status[i] = { 'Unknown': 'columns missing : {}'.format(', '.join(missing_col)) }
            continue
        cred_col_name = list(cred_col.columns)
        deb_col_name = list(deb_col.columns)
        date_col_name = list(date_col.columns)
                
        #* Get solde values
        other_soldes = [table[other_name].str.contains(solde_regx, case=False, na=False) for other_name in table.columns]
        other_solde = other_soldes[0]
        for j in range(1, len(other_soldes)):
            other_solde |= other_soldes[j]
        soldes = table[other_solde.values]
        solde = dict()
        solde['credit'] = [(val.replace(',', '.'), idx)\
            for val, idx in zip(soldes[cred_col_name[0]], soldes[cred_col_name[0]].index) if val != '']
        solde['debit'] = [(val.replace(',', '.'), idx)\
            for val, idx in zip(soldes[deb_col_name[0]], soldes[deb_col_name[0]].index) if val != '']
        
        if len(solde['credit']) < 2 and len(solde['debit']) < 2:
            status[i] = { 'Unknown': 'Not enough solde infos' }
            continue
        
        check_col_n = 'debit' if len(solde['debit']) >= 2 else 'credit'
        first_val, last_val = solde[check_col_n][0], solde[check_col_n][-1]
        sub_table = table[first_val[1]+1:last_val[1]]
        #* Get credit and debit values
        cred_names = cred_col_name + date_col_name
        cred_values = [float(row[0].replace(',', '.'))\
            for row in sub_table[cred_names].to_numpy() if valid_value(row[0], row[1:])]
        cred_val = sum(cred_values)
        deb_names = deb_col_name + date_col_name
        deb_values = [float(row[0].replace(',', '.'))\
            for row in sub_table[deb_names].to_numpy() if valid_value(row[0], row[1:])]
        deb_val = sum(deb_values)
        #* Calc solde final value with table values
        res = round(float(first_val[0]) + (deb_val - cred_val if check_col_n == 'debit' else cred_val - deb_val), 2)
        
        #* Set status depending on calculated solde value matching real final solde value
        if res == float(last_val[0]):
            status[i] = { 'Success': 'Table values match final solde value.' }
        else:
            status[i] = { 'Error' : "table values {} don't match final solde value {}.".format(res, float(last_val[0])) }
    return status