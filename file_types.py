from utils.process_fields import get_agency_information, get_bank_id, get_client_information, get_date
import os

import cv2 as cv
import pandas as pd

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

    def parse_fields(self):
        raise NotImplemented

    def process_text(self):
        raise NotImplemented


class AccountStatements(TesseractDoc):

    def __init__(self, file_path, language, excel_writer, idx=0, debug=False):
        self.debug = debug
        
        self.idx = idx
        self.sheet_name = 'Account statement {}'.format(self.idx)
        
        # Bank infos
        self.information = {
            "Bank name": "N/A",
            "Agency address": "N/A",
            "Agency phone": "N/A",
            "Agency email": "N/A",
            "Consultant phone": "N/A",
            "Consultant email": "N/A",
            "Client full name": "N/A",
            "Client address": "N/A",
            "Date": "N/A"
        }
        
        # Statement rows (date, libellé, montant) sous forme de dataframe
        self.statement_tables = []
        
        self.excel_writer = excel_writer
        self.row = 0

        super().__init__(file_path, language)
    
    def processing(self):
        paths = pdf_to_jpg(self.file_path, self.folder_path)

        for i, path in enumerate(paths):

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
        self.information["Bank name"] = self.bank_utils['name']
        print('Finding bank... DONE')
        
        #* Process fields
        print('Processing fields...\r', end='')
        # Process client information (should be in first page)
        self.information["Client full name"],\
        self.information["Client address"] = get_client_information(
            first_page,
            self.bank_utils,
            self.dicts,
            os.path.join(debug_folder, 'client_info.jpg') if self.debug else None
        )
        # Process Bank information
        self.information["Agency email"],\
        self.information["Agency phone"],\
        self.information["Agency address"] = get_agency_information(
            first_page,
            self.bank_utils,
            self.dicts,
            os.path.join(debug_folder, 'agency_info.jpg') if self.debug else None
        )
        self.information["Date"] = get_date(
            first_page,
            self.bank_utils,
            os.path.join(debug_folder, 'date_info.jpg') if self.debug else None
        )
        self.information["Date"] = self.information["Date"].strftime("%d %B %Y")
        
        infos_df = pd.DataFrame.from_dict(self.information, orient='index')
        infos_df.to_excel(self.excel_writer,
                          sheet_name=self.sheet_name,
                          startcol=0, startrow=self.row)
        self.row += len(self.information) + 2
        print('Processing fields... [DONE]')
        
        #* Process tables
        print('Processing tables...\r', end='')
        page_tables = []
        for i, path in enumerate(self.processed_file_path):
            page_tables += process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
            )
            
        dfs_len = set([len(df.columns) for df in page_tables])
        self.statement_tables = [None] * len(dfs_len)
        for i, df_len in enumerate(dfs_len):
            for df in page_tables:
                if len(df.columns) == df_len:
                    if self.statement_tables[i] is not None:
                        df.columns = self.statement_tables[i].columns
                    self.statement_tables[i] = df if self.statement_tables[i] is None\
                        else pd.concat([self.statement_tables[i], df], ignore_index=True)

        self.statement_tables.sort(key = lambda df: len(df.index), reverse=True)
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            self.row += len(df.index) + 2
        print('Processing tables... [DONE]')
                            

class TaxNotice:

    def __init__(self, file_path, language, debug=False):
        self.debug = debug
        
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

        super().__init__(file_path, language)
    
    def processing(self):
        paths = pdf_to_jpg(self.file_path, self.folder_path)

        for i, path in enumerate(paths):

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
        
        # #* Process fields
        #TODO Process tax notice fields
        # print('Processing fields...\r', end='')
        # # Process client information (should be in first page)
        # self.full_name, self.address = get_client_information(
        #     first_page,
        #     self.bank_utils,
        #     self.dicts,
        #     os.path.join(debug_folder, 'client_info.jpg') if self.debug else None
        # )
        # # Process Bank information
        # self.agency_email, self.agency_phone, self.agency_address = get_agency_information(
        #     first_page,
        #     self.bank_utils,
        #     self.dicts,
        #     os.path.join(debug_folder, 'agency_info.jpg') if self.debug else None
        # )
        # self.statement_date = get_date(
        #     first_page,
        #     self.bank_utils,
        #     os.path.join(debug_folder, 'date_info.jpg') if self.debug else None
        # )
        # print('Processing fields... [DONE]')
        
        #* Process tables
        print('Processing tables...\r', end='')
        for i, path in enumerate(self.processed_file_path):
            if i == 0:
                continue
            self.statement_tables += process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
                semiopen_table=True
            )
        print('Processing tables... DONE')
        
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            df.to_excel(os.path.join(self.folder_path, 'df{}.xlsx'.format(i)))


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