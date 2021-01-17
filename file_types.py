from utils.process_fields import get_agency_information, get_bank_id, get_client_information, get_date
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
        self.full_name = ''
        self.address = ''
        self.statement_date = None

        # Statement rows (date, libellé, montant) sous forme de dataframe
        self.statement_tables = []

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
        
        debug_folder = os.path.join(self.folder_path, 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Load first page as 1D array
        first_page = cv2.imread(self.processed_file_path[0], 0)
        if first_page is None:
            print('Parse fields : error while trying to load {}'.format(self.processed_file_path[0]))
            return
            
        #* Process bank id
        # Bank id should be on first page
        print('Finding bank...\r', end='')
        bank_id = get_bank_id(first_page)
        if bank_id is None:
            print('Error : unknown bank document.')
            return
        # With bank id we can get bank information
        self.bank_utils = get_json_from_file('bank_configs/{}.json'.format(bank_id))
        self.dicts = get_json_from_file('dict.json')
        self.bank_name = self.bank_utils['name']
        print('Finding bank... DONE')
        
        #* Process fields
        print('Processing fields...\r', end='')
        # Process client information (should be in first page)
        self.full_name, self.address = get_client_information(first_page,
                                                              self.bank_utils,
                                                              self.dicts)
        # Process Bank information
        self.agency_email, self.agency_phone, self.agency_address = get_agency_information(first_page,
                                                                                           self.bank_utils,
                                                                                           self.dicts)
        self.statement_date = get_date(first_page, self.bank_utils)
        print('Processing fields... [DONE]')
        
        #* Process tables
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
                # debug_folder=os.path.join(debug_folder, 'page_{}'.format(i))
            )
        print('Processing tables... DONE')
        
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            df.to_excel(os.path.join(self.folder_path, 'df{}.xlsx'.format(i)))
                            

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