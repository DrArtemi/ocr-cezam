import os

import cv2
import pandas as pd
import pytesseract
from pytesseract.pytesseract import Output
from utils.deskew_image import deskew_img
from utils.process_fields import (get_agency_information, get_bank_id,
                                  get_client_information, get_date)
from utils.process_table import process_tables
from utils.utils import get_json_from_file, pdf_to_jpg, process_text, remove_background, save_cv_image

from file_types.file_type import FileType

#TODO Améliorer les résultats

class DocumentIdentite(FileType):
    
    def __init__(self, file_path, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, language, excel_writer, idx=idx, debug=debug)
        
        # Bank infos
        self.information = {
            "Nom": "N/A",
            "Prénom": "N/A",
            "Sexe": "N/A",
            "Date de naissance": "N/A",
            # "Lieu de naissance": "N/A",
            "Numéro d'identité": "N/A",
            # "Truc en bas": "N/A",
            # "Adresse": "N/A",
            "Date validité": "N/A",
            "Date remise": "N/A",
        }
    
    def processing(self):
        extension = self.file_path.split('.')[-1]
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
        
        debug_folder = os.path.join(self.folder_path, 'di_debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        fields = {
            "Numéro d'identité": [ ['carte', 'nationale'], False, 0 ],
            "Nom": [ ['nom'], False, 0 ],
            "Prénom": [ ['prénom'], False, 0 ],
            "Sexe": [ ['sexe'], False, 0 ],
            "Date de naissance": [ ['né'], False, 1 ],
            "Date validité": [ ['carte', 'valable'], False, 0 ],
            "Date remise": [ ['délivrée', 'le'], False, 0 ]
        }
        
        for p_file in self.processed_file_path:
            img = cv2.imread(p_file, 0)

            if img is None:
                print("Error : Can't load {}".format(p_file))
                return False
            
            cleaned = remove_background(img)        
            
            if debug_folder is not None:
                cv2.imwrite(os.path.join(debug_folder, 'test.jpg'), cleaned)
                
            text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, lang='fra')
            text, conf, bb = process_text(text_data)
            
            for key in fields:
                if not fields[key][1]:
                    self.information[key], fields[key][1] = self.get_field(text, fields[key][0], idx=fields[key][2])
        
        infos_df = pd.DataFrame.from_dict(self.information, orient='index')
        infos_df.to_excel(self.excel_writer,
                          sheet_name=self.sheet_name,
                          startcol=0, startrow=self.row)
        self.row += len(self.information) + 2
        return True

    def get_field(self, text, fields, idx=0):
        for row in text:
            res = sum([any([f in w.lower() for w in row]) for f in fields]) == len(fields)
            if res:
                return self.get_idx_term(row, idx)
        return 'N/A', False
    
    @staticmethod
    def get_idx_term(row, idx):
        cnt = 0
        for i in range(len(row)):
            if ':' in row[i]:
                if idx == cnt:
                    return row[i+1], True
                else:
                    cnt += 1
        return 'N/A', False
