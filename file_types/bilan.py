import os

import cv2
from numpy.core.fromnumeric import resize
import numpy as np
import pandas as pd
import pytesseract
from pytesseract.pytesseract import Output
from utils.deskew_image import deskew_img
from utils.process_fields import (get_agency_information, get_bank_id,
                                  get_client_information, get_date)
from utils.process_table import process_tables
from utils.utils import get_json_from_file, pdf_to_jpg, process_text, remove_background, save_cv_image

from file_types.file_type import FileType


class Bilan(FileType):
    
    def __init__(self, file_path, doc_type, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, doc_type, language, excel_writer, idx=idx, debug=debug)
        
        self.cwd = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        
        # Bank infos
        self.information = {
            "Siret": "N/A",
            "APE": "N/A",
            "Client full name": "N/A",
            "Client address": "N/A",
            "Date": "N/A"
        }
        self.numbers = {
            '1': ['010', '014', '028', '040', '044', '050', '060', '064', '068', '072', '080', '084', '088', '092', '096',
                  '110', '193', '197', '199', '195', '182', '184', '209', '215', '217', '229', '243', '259', '316', '318',
                  '322', '324', '247', '248', '330', '342', '344', '346', '350', '352', '354', '356', '360', '366', '368',
                  '370', '372', '374', '376', '378', '380', '399', '400', '402', '404', '406', '410', '412', '414', '416',
                  '420', '422', '424', '426', '430', '432', '434', '436', '440', '442', '444', '446', '450', '452', '454',
                  '456', '460', '462', '464', '466', '470', '472', '474', '476', '480', '482', '484', '486', '490', '492',
                  '494', '496', '500', '502', '504', '506', '510', '512', '514', '516', '520', '522', '524', '526', '530',
                  '532', '534', '536', '540', '542', '544', '546', '550', '552', '554', '556', '560', '562', '564', '566',
                  '570', '572', '574', '576', '578', '580', '582', '584', '586', '588', '590', '592', '593', '596', '600',
                  '602', '604', '606', '610', '612', '614', '616', '620', '622', '624', '626', '630', '632', '634', '636',
                  '640', '642', '644', '646', '650', '652', '654', '656', '660', '662', '664', '666', '680', '682', '684',
                  '686', '700', '705', '710', '715', '720', '725', '730', '735', '740', '745', '750', '755', '760', '765',
                  '770', '775', '780', '800', '804', '810', '814', '818', '820', '824', '828', '830', '834', '838', '840',
                  '844', '848', '850', '844', '850', '854', '860', '870', '900', '910', '920', '930', '950', '960'],
            '2': ['120', '124', '126', '130', '132', '134', '136', '140', '142', '154', '156', '164', '166', '172', '174',
                  '176', '180', '210', '214', '218', '222', '224', '226', '230', '232', '234', '236', '238', '240', '242',
                  '244', '250', '252', '254', '256', '262', '264', '270', '280', '290', '264', '300', '306', '310', '312'],
            '3': ['012', '016', '030', '042', '048', '052', '062', '066', '070', '074', '082', '086', '090', '094', '098',
                  '112'],
        }
        self.number_values = []
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
        
        for f, p_file in enumerate(self.processed_file_path):
            img = cv2.imread(p_file, 0)

            if img is None:
                print("Error : Can't load {}".format(p_file))
                return False
            
            resizing = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cleaned = cv2.threshold(resizing, 170, 255, cv2.THRESH_BINARY)[1]
    
            if debug_folder is not None:
                cv2.imwrite(os.path.join(debug_folder, f'cleaned_{f}.jpg'), cleaned)
            
            text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, lang='fra')
            text, conf, bb = process_text(text_data)
            
            check = False
            for i, row in enumerate(text):
                if 'dgfip' in ' '.join(row).lower():
                    check = True
            if check:
                statement_tables, tables, tables_bb = process_tables(
                    p_file,
                    arrange_mode=0,
                    debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
                    clean_cells=False,
                    num_columns=[]
                )
                self.statement_tables += statement_tables
                
                self.number_values += self.get_numbers_values(tables, tables_bb)
                
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            self.row += len(df.index) + 2
        print(self.number_values)
        print(np.array(self.number_values))
        df = pd.DataFrame.from_records(self.number_values)
        df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
        self.row += len(df.index) + 2
        print('Processing tables... [DONE]')
        return True
    
    def get_numbers_values(self, tables, tables_bb):
        tables_numbers = []
        for i, table_content in enumerate(tables):
            tables_numbers.append([])
            for j, row in enumerate(table_content):
                for k, word in enumerate(row):
                    if word.replace('\n', '').isdigit() and\
                        len(tables_bb[i][j][k]) > 0 and\
                        35 < tables_bb[i][j][k][2] < 45:
                            tables_numbers[-1].append([j, k, word.replace('\n', '')])
                    # print(repr(word))
        numbers_values = []
        for i, tn in enumerate(tables_numbers):
            nv = []
            for n in tn:
                for nb_values in self.numbers:
                    if n[2] in self.numbers[nb_values]:
                        pos = n[1]+1
                        nv.append([n[2]] + [w.replace('\n', '') for w in tables[i][n[0]][pos:pos+int(nb_values)]])
            numbers_values += nv
                        
        return numbers_values
  
