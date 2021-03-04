import os

import cv2
from numpy.core.fromnumeric import resize
from numpy.lib.shape_base import split
import pandas as pd
import pytesseract
from pytesseract.pytesseract import Output
from utils.deskew_image import deskew_img
from utils.utils import pdf_to_jpg, process_text, remove_background, save_cv_image

from file_types.file_type import FileType
import numpy as np


class DocumentIdentite(FileType):
    
    def __init__(self, file_path, doc_type, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, doc_type, language, excel_writer, idx=idx, debug=debug)
        
        self.cwd = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        
        # Bank infos
        self.information = {
            "Nom": "N/A",
            "Prénom": "N/A",
            "Sexe": "N/A",
            "Date de naissance": "N/A",
            "Lieu de naissance": "N/A",
            "Numéro d'identité": "N/A",
            "Adresse": "N/A",
            "Date validité": "N/A",
            "Date remise": "N/A",
            "Lieu remise": "N/A",
            "MRZ ligne 1": "N/A",
            "MRZ ligne 2": "N/A"
        }
    
    def processing(self):
        extension = self.file_path.split('.')[-1].lower()
        del_original = True
        print(extension)
        if extension == 'pdf':
            paths = pdf_to_jpg(self.file_path, self.folder_path)
        elif extension in ['jpg', 'jpeg']:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            paths = [self.file_path]
            del_original = False
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
            self.processed_file_path.append(save_cv_image(img, path, 'jpg', del_original=del_original))
            
        if len(paths) == 0:
            print('Error: no pages found in {}'.format(self.file_path))
            return False
        return True

    def parse_fields(self):
        
        debug_folder = os.path.join(self.folder_path, 'di_debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Each field = [ Trigger words, Already processed field, Row field pos,  Value is all the line ]
        fields = {
            "Numéro d'identité": [ ['carte', 'nationale'], False, 0, False ],
            "Nom": [ ['nom'], False, 0, True ],
            "Prénom": [ ['prénom'], False, 0, True ],
            "Sexe": [ ['sexe'], False, 0, False ],
            "Adresse": [ ['adresse'], False, 0, True ],
            "Date de naissance": [ ['né'], False, 1, True ],
            "Lieu de naissance": [ ['à'], False, 0, True ],
            "Date remise": [ ['délivrée', 'le'], False, 0, True ],
            "Date validité": [ ['carte', 'valable'], False, 0, True ],
            "Lieu remise": [ ['par'], False, 0, True ],
        }
        
        for f, p_file in enumerate(self.processed_file_path):
            img = cv2.imread(p_file, 0)

            if img is None:
                print("Error : Can't load {}".format(p_file))
                return False
            
            cleaned2 = self.custom_remove_background(img, kernel=(2, 4))
            
            if debug_folder is not None:
                cv2.imwrite(os.path.join(debug_folder, f'cleaned2_{f}.jpg'), cleaned2)
                            
            text_data = pytesseract.image_to_data(cleaned2, output_type=Output.DICT, lang='fra')
            text, conf, bb = process_text(text_data)

            self.information['MRZ ligne 1'], self.information['MRZ ligne 2'] = self.get_mrz(text)
            if self.information['MRZ ligne 1'] != 'N/A' and self.information['MRZ ligne 2'] != 'N/A':
                fields = self.fill_with_mrz(fields)
            for key in fields:
                if not fields[key][1]:
                    self.information[key], fields[key][1] = self.get_field(text,
                                                                           fields[key][0],
                                                                           idx=fields[key][2],
                                                                           all_line=fields[key][3])
            
        
        infos_df = pd.DataFrame.from_dict(self.information, orient='index')
        infos_df.to_excel(self.excel_writer,
                          sheet_name=self.sheet_name,
                          startcol=0, startrow=self.row)
        self.row += len(self.information) + 2
        return True
    
    def fill_with_mrz(self, fields):
        mrz_information = {
            "Nom": self.information['MRZ ligne 1'][5:30].replace('<', ''),
            "Prénom": ' '.join(self.information['MRZ ligne 2'][13:27].replace('<', ' ').split()),
            "Sexe": self.information['MRZ ligne 2'][34],
            "Date de naissance": self.mrz_date_to_date(self.information['MRZ ligne 2'][27:33]),
            "Date remise": self.information['MRZ ligne 2'][2:4] + '.' + self.information['MRZ ligne 2'][0:2],
            "Lieu remise": self.information['MRZ ligne 1'][30:32],
        }
        
        for info in mrz_information:
            if self.information[info] == 'N/A':
                self.information[info] = mrz_information[info]
                fields[info][1] = True
        return fields

    @staticmethod
    def mrz_date_to_date(date):
        splited = [date[i:i+2] for i in range(0, len(date), 2)]
        new_date = splited[2] + '.' + splited[1] + '.' + splited[0]
        return new_date

    def get_mrz(self, text):
        mrz = []
        for row in text:
            stacked_row = ''.join(row)
            if len(stacked_row) == 36 and '<' in stacked_row:
                mrz.append(stacked_row)
        if len(mrz) > 1:
            return mrz[0], mrz[1]
        return 'N/A', 'N/A'

    def get_field(self, text, fields, idx=0, all_line=False):
        for row in text:
            res = sum([any([f in w.lower() for w in row]) for f in fields]) == len(fields)
            if res:
                # print(fields, row)
                return self.get_idx_term(row, idx, all_line)
        return 'N/A', False
    
    @staticmethod
    def get_idx_term(row, idx, all_line=False):
        cnt = 0
        for i in range(len(row)):
            if ':' in row[i]:
                if idx == cnt and i+1 < len(row):
                    return (' '.join(row[i+1:]) if all_line else row[i+1]), True
                else:
                    cnt += 1
        return 'N/A', False

    @staticmethod
    def custom_process(img):
        return img
    
    @staticmethod
    def custom_remove_background(img, kernel=(2, 1), iterations=1):
        cv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        border = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        dilation = cv2.dilate(resizing, cv_kernel, iterations=iterations)
        erosion = cv2.erode(dilation, cv_kernel, iterations=iterations)
        #* Image cleaning
        # Close small dots
        clean_dots = cv2.morphologyEx(src=erosion, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
        # Resharpen our text by making binary img
        cleaned = cv2.threshold(clean_dots, 170, 255, cv2.THRESH_BINARY)[1]
        return cleaned
