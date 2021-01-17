import pytesseract
import errno
import csv
import os

import cv2 as cv
import numpy as np
import pandas as pd

from PIL import Image
from pytesseract.pytesseract import Output

from utils.process_table import process_tables
from utils.deskew_image import deskew_img
from utils.utils import *


class TesseractDoc:

    def __init__(self, file_path, language):
        self.file_path = file_path
        self.folder_path = '.'.join(self.file_path.split('.')[:-1])
        self.language = language

        self.processed_file_path = []

        self.file_text_data = []
        self.images_size = []
    
    #! Pas utilisé pour l'instant
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

        # Statement rows (date, libellé, montant) sous forme de dataframe
        self.statement_tables = []
        self.columns = None
        self.v_lines = None
        self.h_lines = None

        # Image list of pdf
        self.images = []

        super().__init__(file_path, language)
    
    def processing(self):
        paths = pdf_to_jpg(self.file_path, self.folder_path)

        for path in paths:

            # Convert tiff to cv2 img
            img = cv.imread(path, 0)
            
            if img is None:
                print('Error while trying to load {}'.format(path))
                continue

            # Rotate img
            img = deskew_img(img)

            # Save image to jpg and remove tiff
            self.processed_file_path.append(save_cv_image(img, path, 'jpg', del_original=True))
    
    def parse_fields(self):
        #TODO Ici l'idée ce serait pour chaque chaque field de :
        #TODO   - Zoom sur la zone passée dans le json
        #TODO   - Clean la zone (genre si y a un fond ou des points)
        #TODO   - Extraire le texte
        #TODO   - Traiter le texte pour en sortir que l'info attendue
        #TODO En plus de ça il faut extraire le(s) tableau(x) et leurs données
        #TODO Et si tout ça c'est bien fait ben jsuis juste un bg
        
        debug_folder = os.path.join(self.folder_path, 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # * Process tables, récupère une liste de dataframes
        #? Si j'ai deux dataframes avec le même nombre de colonnes je fais quoi ?
        #?  - Je les merge
        #?  - Je les laisse séparés
        #? Pour le reste, j'en fais un excel ? Plusieurs tableaux dans le même doc ?
        #? Une sheet par table ? Un excel par table ?
        print('Processing tables...\r', end='')
        for i, path in enumerate(self.processed_file_path):
            self.statement_tables += process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i))
            )
        print('Processing tables... DONE')
        
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            df.to_excel(os.path.join(self.folder_path, 'df{}.xlsx'.format(i)))
                
    
    def parse_fields_old(self):
        for i, text_data in enumerate(self.file_text_data):
            text, conf, bb = self.process_text(text_data)
            
            # Get information available on the first page
            if i == 0:
                bank_id = get_bank_id(text)
                if bank_id is None:
                    print('Error : unknown bank document.')
                    return
                self.bank_utils = get_json_from_file('bank_configs/{}.json'.format(bank_id))
                self.dicts = get_json_from_file('dict.json')
                self.bank_name = self.bank_utils['name']
                self.last_name, self.first_name = get_client_name(text)
                self.agency_address, self.address = get_addresses(text, bb, self.images_size[i], self.bank_utils, self.dicts)
                self.agency_phone = get_agency_phone(text, bb, self.images_size[i], self.bank_utils, self.dicts)
                self.agency_email = get_agency_email(text, bb, self.images_size[i], self.bank_utils)
                _, self.statement_month, self.statement_year = get_date(text, bb, self.images_size[i], self.bank_utils)
            self.process_table(i, text, bb)
            # exit()

            if self.statement_row is not None:
                print('Rows shape : {}'.format(self.statement_row.shape))

            
            save_bb_image(self.processed_file_path[i], bb, self.v_lines, self.h_lines)

            # return
    
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

        inv = cv.bitwise_not(gray)
        ret1, th1 = cv.threshold(inv, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        kernel_len = np.array(img).shape[1] // 100
        ver_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_len))
        hor_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_len, 1))
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        
        image_1 = cv.erode(th1, ver_kernel, iterations=3)
        vertical_lines = cv.dilate(image_1, ver_kernel, iterations=3)
        
        image_2 = cv.erode(th1, hor_kernel, iterations=3)
        horizontal_lines = cv.dilate(image_2, hor_kernel, iterations=3)
        
        img_vh = cv.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        img_vh = cv.erode(~img_vh, kernel, iterations=2)
        thresh, img_vh = cv.threshold(img_vh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        img_vh = 255 - img_vh
        
        min_len = gray.shape[0] * 0.05  # Taille minimale d'une ligne
        max_gap = 20  # Ecrat maximal pour considérer que c'est la même ligne
        lines = cv.HoughLinesP(img_vh, rho=1, theta=(np.pi / 180), threshold=50,
                               minLineLength=min_len, maxLineGap=max_gap)
        
        print('-- Infos --')
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
        self.v_lines = overlapping_lines_filter(v_lines, 0)
        self.h_lines = overlapping_lines_filter(h_lines, 1)
        self.v_lines, self.h_lines = irrelevant_lines_filter(self.v_lines, self.h_lines)

        # On récupère les extrémités du tableau (top, bottom, left, right)
        if len(self.h_lines) < 2 or len(self.v_lines) < 2:
            return False
        table_shape = (self.h_lines[0][1], self.h_lines[-1][1], self.v_lines[0][0], self.v_lines[-1][0])
        if self.columns is None:
            self.columns = get_table_columns(text, bb, table_shape, self.v_lines)
        if self.columns is None or len(self.v_lines) - 1 != len(self.columns):
            print('No valid table on this page :', len(self.v_lines), len(self.columns) if self.columns is not None else None)
            return False
        data = []
        for i in range(len(text)):
            row = [''] * len(self.columns)
            cnt = 0
            for j in range(len(self.v_lines) - 1):
                words = []
                for h, w in enumerate(text[i]):
                    if self.v_lines[j][0] < bb[i][h][0] < self.v_lines[j+1][0] and\
                        table_shape[0] < bb[i][h][1] < table_shape[1]:
                        words.append(w)
                if len(words) > 0:
                    row[j] = ' '.join(words)
                    cnt += 1
            if cnt > 0:
                data.append(row)
        
        if self.statement_row is None:
            self.statement_row = pd.DataFrame(data, columns=self.columns)
        else:
            tmp_statement = pd.DataFrame(data, columns=self.columns)
            self.statement_row = self.statement_row.append(tmp_statement, ignore_index=True)
        return True
        

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