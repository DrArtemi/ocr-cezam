import pytesseract
import errno
import csv
import os

import cv2 as cv
import numpy as np
import pandas as pd

from PIL import Image
from pytesseract.pytesseract import Output

from utils import *


class TesseractDoc:

    def __init__(self, file_path, language):
        self.file_path = file_path
        self.folder_path = '.'.join(self.file_path.split('.')[:-1])
        self.language = language

        self.processed_file_path = []

        self.file_text_data = []
        self.images_size = []
    
    def extract_text(self, save_file=False):
        self.file_text_data = []
        self.images_size = []

        if self.processed_file_path is None:
            print('Error: {}: Processed file not found.'.format(self.processed_file_path))
        
        custom_config = r'--oem 3 --psm 4'
        for i, processed_fp in enumerate(self.processed_file_path):
            img = Image.open(processed_fp)
            self.images_size.append(img.size)
            res = pytesseract.image_to_data(img,
                                            output_type=Output.DICT,
                                            config=custom_config,
                                            lang=self.language)
            self.file_text_data.append(res)

    def parse_fields(self):
        raise NotImplemented

    def process_text(self):
        raise NotImplemented


class AccountStatements(TesseractDoc):

    def __init__(self, file_path, language):
        # Bank infos
        self.bank_name = ''
        self.agency_address = ''
        self.agency_phone = ''
        self.agency_email = ''
        self.consultant_phone = ''
        self.consultant_email = ''

        # Cient infos
        self.last_name = ''
        self.first_name = ''
        self.address = ''
        self.statement_year = ''
        self.statement_month = ''

        # Statement rows (date, libellé, montant)
        self.statement_row = None
        self.columns = None

        # Image list of pdf
        self.images = []

        super().__init__(file_path, language)
    
    def processing(self):
        paths = split_pdf_pages(self.file_path, self.folder_path)

        for path in paths:
            # Convert pdf to tiff to be able to process it
            tiff_path = pdf_to_tiff(path)

            # Convert tiff to cv2 img
            img = cv.imread(tiff_path, 0)

            # Remove noise background
            img = remove_dot_background(img, kernel=(5, 5))

            # Save image to jpg and remove tiff
            self.processed_file_path.append(save_cv_image(img, tiff_path, 'jpg', del_original=True))
    
    def process_text(self, text_data):
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
                bb_list.append((text_data['left'][i],
                                text_data['top'][i],
                                text_data['width'][i],
                                text_data['height'][i]))
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
    
    def process_table(self, idx, text, bb):
        file = self.processed_file_path[idx]
        img = cv.imread(file)
        # Extraction des lignes du tableau
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 150, apertureSize=3)
        min_len = edges.shape[1] * 0.75  # Taille minimale d'une ligne
        max_gap = 6  # Ecrat maximal pour considérer que c'est la même ligne
        lines = cv.HoughLinesP(edges, rho=1, theta=(np.pi / 180), threshold=50,
                               minLineLength=min_len, maxLineGap=max_gap)
        
        # Séparation des lignes verticales et horizontales
        v_lines = []
        h_lines = []
        for i in range(len(lines)):
            l = lines[i][0]
            if is_line_vertical(l):
                v_lines.append(l)
            elif is_line_horizontal(l):
                h_lines.append(l)
        
        # On filtre les doublons
        v_lines = overlapping_lines_filter(v_lines, 0)
        h_lines = overlapping_lines_filter(h_lines, 1)

        # On récupère les extrémités du tableau (top, bottom, left, right)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return False
        table_shape = (h_lines[0][1], h_lines[-1][1], v_lines[0][0], v_lines[-1][0])
        if self.columns is None:
            self.columns = get_table_columns(text, bb, table_shape, v_lines)
        if len(v_lines) - 1 != len(self.columns):
            print('No valid table on this page :', len(v_lines), len(self.columns))
            return False
        data = []
        for i in range(len(text)):
            row = [''] * len(self.columns)
            cnt = 0
            for j in range(len(v_lines) - 1):
                words = []
                for h, w in enumerate(text[i]):
                    if v_lines[j][0] < bb[i][h][0] < v_lines[j+1][0] and\
                        table_shape[0] < bb[i][h][1] < table_shape[1]:
                        words.append(w)
                if len(words) > 0:
                    row[j] = ' '.join(words)
                    cnt += 1
            if cnt > 0:
                data.append(row)
        
        # print('Coucou')
        if self.statement_row is None:
            # print('Coucou 1')
            self.statement_row = pd.DataFrame(data, columns=self.columns)
        else:
            # print('Coucou 2')
            tmp_statement = pd.DataFrame(data, columns=self.columns)
            # print(tmp_statement)
            self.statement_row = self.statement_row.append(tmp_statement, ignore_index=True)
        return True

       
    def parse_fields(self):
        for i, text_data in enumerate(self.file_text_data):
            text, conf, bb = self.process_text(text_data)
            
            # Get information available on the first page
            if i == 0:
                # TODO : implémenter les trucs la
                bank_id = get_bank_id(text)
                self.bank_utils = get_json_from_file('bank_configs/{}.json'.format(bank_id))
                self.dicts = get_json_from_file('dict.json')
                self.bank_name = self.bank_utils['name']
                self.last_name, self.first_name = get_client_name(text)
                self.agency_address, self.address = get_addresses(text, bb, self.images_size[i], self.bank_utils, self.dicts)
                self.agency_phone = get_agency_phone(text, bb, self.images_size[i], self.bank_utils, self.dicts)
                self.agency_email = get_agency_email(text, bb, self.images_size[i], self.bank_utils)
                _, self.statement_month, self.statement_year = get_date(text, bb, self.images_size[i], self.bank_utils)
            self.process_table(i, text, bb)
            print('Rows shape : {}'.format(self.statement_row.shape))

            
            save_bb_image(self.processed_file_path[i], bb)
            # for j in range(len(text)):
                # print(text[j])
                # print(conf)
                # print(bb)
                # print('-----------------------------------------------')
            # return
        

class TaxNotice:

    def __init__():
        pass


class IdentityDocument(TesseractDoc):

    def __init__(self, file_path, language):

        # Names
        self.last_name = ""
        self.birth_name = ""
        self.first_name = ""

        # Birth
        self.birth_date = ""
        self.birth_place = ""

        # Doc infos
        self.type = ""
        self.issued_date = ""
        self.expiration_date = ""

        super().__init__(file_path, language)