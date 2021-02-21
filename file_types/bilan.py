import os

import cv2
import pandas as pd
from utils.deskew_image import deskew_img
from utils.process_fields import (get_agency_information, get_bank_id,
                                  get_client_information, get_date)
from utils.process_table import process_tables
from utils.utils import get_json_from_file, pdf_to_jpg, save_cv_image

from file_types.file_type import FileType

#TODO Implémenter le bilan


class Bilan(FileType):
    
    def __init__(self, file_path, doc_type, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, doc_type, language, excel_writer, idx=idx, debug=debug)
        
        # Bank infos
        self.information = {
            "Siret": "N/A",
            "APE": "N/A",
            "Client full name": "N/A",
            "Client address": "N/A",
            "Date": "N/A"
        }
        self.statement_tables = []
        
    def processing(self):
        extension = self.file_path.split('.')[-1].lower()
        if extension == 'pdf':
            paths = pdf_to_jpg(self.file_path, self.folder_path)
        elif extension in ['jpg', 'jpeg']:
            paths = [self.file_path]
        else:
            print('Error: {} is not a valid PDF or JPG file'.format(self.file_path))
            return False

        for path in paths:

            # Convert tiff to cv2 img
            img = cv2.imread(path, 0)
            
            if img is None:
                print('Error while trying to load {}'.format(path))
                continue

            # Rotate img
            img = deskew_img(img)

            # Save image to jpg and remove tiff
            self.processed_file_path.append(save_cv_image(img, path, 'jpg', del_original=True))
            
        if len(paths) == 0:
            print('Error: no pages found in {}'.format(self.file_path))
            return False
        return True

    def parse_fields(self):
        
        debug_folder = os.path.join(self.folder_path, 'b_debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Load first page as 1D array
        first_page = cv2.imread(self.processed_file_path[0], 0)
        if first_page is None:
            print("Error : Can't load {}".format(self.processed_file_path[0]))
            return False
            
        #* Process bank id
        # Bank id should be on first page
        print('Finding bank...\r', end='')
        bank_id = get_bank_id(first_page)
        if bank_id is None:
            print('Error : unknown bank document.')
            return False
        
        # infos_df = pd.DataFrame.from_dict(self.information, orient='index')
        # infos_df.to_excel(self.excel_writer,
        #                   sheet_name=self.sheet_name,
        #                   startcol=0, startrow=self.row)
        # self.row += len(self.information) + 2
        # print('Processing fields... [DONE]')
        
        #* Process tables
        print('Processing tables...\r', end='')
        page_tables = []
        for i, path in enumerate(self.processed_file_path):
            if i == 0:
                continue
            self.statement_tables += process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
            )
            # if i == 3:
            #     break

        self.statement_tables.sort(key = lambda df: len(df.index), reverse=True)
        # tables_status = self.check_solde()
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            # status_df = pd.DataFrame.from_dict(tables_status[i], orient='index', columns=['Description'])
            # status_df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            # self.row += 2
            df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            self.row += len(df.index) + 2
        print('Processing tables... [DONE]')
        return True