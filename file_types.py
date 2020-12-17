import pytesseract
import errno
import os

import cv2 as cv

from PIL import Image

from utils import remove_dot_background, pdf_to_tiff, save_cv_image


class TesseractDoc:

    def __init__(self, file_path, language):
        self.file_path = file_path
        self.language = language

        self.processed_file_path = file_path

        self.file_text = None
        self.file_text_data = None
        self.file_text_bb = None
    
    
    def extract_text(self, bb=False, data=False):
        if self.processed_file_path is None:
            print('Error: {}: Processed file not found.'.format(self.processed_file_path))
        
        self.file_text = pytesseract.image_to_string(Image.open(self.processed_file_path), lang=self.language)
        if bb:
            self.file_text_bb = pytesseract.image_to_boxes(Image.open(self.processed_file_path), lang=self.language)
        if data:
            self.file_text_data = pytesseract.image_to_data(Image.open(self.processed_file_path), lang=self.language)

    def parse_fields(self):
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
        self.statement_row = []

        super().__init__(file_path, language)
    
    def processing(self):
        # Convert pdf to tiff to be able to process it
        tiff_path = pdf_to_tiff(self.file_path)

        # Convert tiff to cv2 img
        img = cv.imread(tiff_path, 0)

        # Remove noise background
        img = remove_dot_background(img, kernel=(4, 4))

        # Save image to jpg and remove tiff
        self.processed_file_path = save_cv_image(img, tiff_path, 'jpg', del_original=True)





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