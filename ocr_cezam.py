from file_types.bulletin_paie import BulletinPaie
from file_types.tableau_amortissement import TableauAmortissement
from file_types.document_identite import DocumentIdentite
from file_types.bilan import Bilan
from file_types.avis_imposition import AvisImposition
from file_types.releve_bancaire import ReleveBancaire
import locale
import argparse
from utils.utils import get_json_from_file
import pytesseract

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("-config", type=str, default=None, help="path to document", required=True)
    
    parser.add_argument("-excel-path", type=str, default='processed.xlsx', help="path where excel will be saved")
    parser.add_argument("-lang", type=str, default='fra', help="document language")

    return parser.parse_args()


def set_locale(language):
    lang_tab = {
        'fra': 'fr_FR.utf8',
        'eng': 'en_US.utf8'
    }
    locale.setlocale(locale.LC_ALL, lang_tab[language])


def get_image(path, doc_type, language, excel_writer, idx, debug):
    switcher = {
        "releveBancaire": ReleveBancaire,
        "avisImposition": AvisImposition,
        "liasseFiscale": Bilan,
        "documentIdentite": DocumentIdentite,
        "tableauAmortissement": TableauAmortissement,
        "bulletinPaie": BulletinPaie
    }
    image_class = switcher.get(doc_type)
    return image_class(path, doc_type, language, excel_writer, idx=idx, debug=debug)


if __name__ == '__main__':
    # Get arguments
    args = parse_args()
    
    print('*******************************************')
    print('TESSERACT VERSION : {}'.format(pytesseract.get_tesseract_version()))
    print('*******************************************')
        
    if args.config is None:
        print('Error: You must give a path to a document. Use python ocr_cezam.py -h for more information.')

    set_locale(args.lang)
        
    config = get_json_from_file(args.config)
    excel_writer = pd.ExcelWriter(args.excel_path, engine='xlsxwriter')
        
    for document_type in config:
        for i, document in enumerate(config[document_type]):
            print('Processing {} {}... ({})'.format(document_type, i, document))
            # Get image class
            try:
                image = get_image(document, document_type, args.lang, excel_writer, i, True)
                # Process image (create folder, separate pdf pages to different images, process images)
                if not image.processing():
                    print('Error while trying to process {}, moving on to the next document'.format(document))
                    continue
            except:
                print(f'Error, could not process image for {document}')
                continue
            # image.extract_text(save_file=True)
            try:
                if not image.parse_fields():
                # Save Excel Writer
                    print('Error while trying to parse fields of {}, moving on to the next document'.format(document))
                    continue
            except:
                print(f'Error, could not parse fields for {document}')
    excel_writer.save()
    print('Processing ended, leaving !')
