import locale
import argparse
from utils.utils import get_json_from_file
import pytesseract

import pandas as pd

from file_types import IdentityDocument, TaxNotice, AccountStatements


def parse_args():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("-config", type=str, default=None, help="path to document", required=True)
    
    parser.add_argument("-lang", type=str, default='fra', help="document language")

    return parser.parse_args()


def set_locale(language):
    lang_tab = {
        'fra': 'fr_FR.utf8',
        'eng': 'en_US.utf8'
    }
    locale.setlocale(locale.LC_ALL, lang_tab[language])


def account_statements(path, language, excel_writer, idx, debug):
    return AccountStatements(path, language, excel_writer, idx=idx, debug=debug)


def tax_notice(path, language, excel_writer, idx, debug):
    return TaxNotice(path, language, excel_writer, idx=idx, debug=debug)


def identity_document(path, language):
    return IdentityDocument(path, language=language)


def get_image(path, doc_type, language, excel_writer, idx, debug):
    switcher = {
        "account_statements": account_statements,
        "tax_notices": tax_notice,
        "identity_documents": identity_document
    }
    image_class = switcher.get(doc_type)
    return image_class(path, language, excel_writer, idx, debug)


if __name__ == '__main__':
    # Get arguments
    args = parse_args()
    
    #TODO Pourvoir passer un fichier de config :
    #TODO   - Liste des relevés de comptes (1 sheet/doc avec la date)
    #TODO   - Liste des avis d'imposition
    #TODO   - Liste des documents d'identité

    print('*******************************************')
    print('TESSERACT VERSION : {}'.format(pytesseract.get_tesseract_version()))
    print('*******************************************')

    if args.config is None:
        print('Error: You must give a path to a document. Use python ocr_cezam.py -h for more information.')

    set_locale(args.lang)
        
    config = get_json_from_file(args.config)
    excel_path = config["name"] + '.xlsx'
    excel_writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    
    for i, acc_stat in enumerate(config["account_statements"]):
        print('Processing account statement {}...'.format(i))
        # Get image class
        image = get_image(acc_stat, "account_statements", args.lang, excel_writer, i, True)
        # Process image (create folder, separate pdf pages to different images, process images)
        image.processing()
        # image.extract_text(save_file=True)
        image.parse_fields()
        
    for i, tax_notice in enumerate(config["tax_notices"]):
        print('Processing tax notice {}...'.format(i))
        # Get image class
        image = get_image(tax_notice, "tax_notices", args.lang, excel_writer, idx=i, debug=True)
        # Process image (create folder, separate pdf pages to different images, process images)
        image.processing()
        # image.extract_text(save_file=True)
        image.parse_fields()
    
    excel_writer.save()
