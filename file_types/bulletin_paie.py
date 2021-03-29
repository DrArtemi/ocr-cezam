import os

import cv2
import pandas as pd
from utils.deskew_image import deskew_img
from utils.process_fields import (get_agency_information, get_bank_id,
                                  get_client_information, get_date)
from utils.process_table import process_tables
from utils.utils import get_json_from_file, pdf_to_jpg, save_cv_image

from file_types.file_type import FileType


class BulletinPaie(FileType):
    
    def __init__(self, file_path, doc_type, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, doc_type, language, excel_writer, idx=idx, debug=debug)
        
        self.cwd = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        
        # Bank infos
        self.information = {
            "Client full name": "N/A",
            "Client address": "N/A",
            "Date": "N/A"
        }
    
    def processing(self):
        extension = self.file_path.split('.')[-1].lower()
        if extension == 'pdf':
            paths = pdf_to_jpg(self.file_path, self.folder_path)
        elif extension in ['jpg', 'jpeg']:
            paths = [self.file_path]
        else:
            print('Error: {} is not a valid PDF or JPG file'.format(self.file_path))
            return False

        if paths is None:
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
        
        debug_folder = os.path.join(self.folder_path, 'rb_debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Load first page as 1D array
        first_page = cv2.imread(self.processed_file_path[0], 0)
        if first_page is None:
            print("Error : Can't load {}".format(self.processed_file_path[0]))
            return False
        
        #* Process fields
        print('Processing fields...\r', end='')
        self.dicts = get_json_from_file(os.path.join(self.cwd, 'dict.json'))
        # Process client information (should be in first page)
        self.information["Client full name"],\
        self.information["Client address"] = get_client_information(
            first_page,
            { "client_info": [ [0.41, 0.96], [0.17, 0.27] ] },
            self.dicts,
            os.path.join(debug_folder, 'client_info.jpg') if self.debug else None
        )
        self.information["Date"] = get_date(
            first_page,
            {
                "date_info": [ [0.0, 0.4], [0.07, 0.1] ],
                "date_format": "%d/%m/%Y",
            },
            os.path.join(debug_folder, 'date_info.jpg') if self.debug else None
        )
        if self.information["Date"] != 'N/A':
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
            statement_tables, tables, tables_bb = process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
            )
            page_tables += statement_tables
            
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
        return True
    