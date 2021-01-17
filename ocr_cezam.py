import locale
import argparse
import pytesseract

from file_types import IdentityDocument, TaxNotice, AccountStatements


def parse_args():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("-path", type=str, default=None, help="path to document", required=True)

    parser.add_argument("-type", type=int, default=0,
                        help="type of the document [0 = relevé de compte, 1 = avis d'imposition, 2 = piece d'identite]",
                        required=True)
    parser.add_argument("-lang", type=str, default='fra', help="document language")

    return parser.parse_args()


def set_locale(language):
    lang_tab = {
        'fra': 'fr_FR.utf8',
        'eng': 'en_US.utf8'
    }
    locale.setlocale(locale.LC_ALL, lang_tab[language])


def account_statements(path, language):
    return AccountStatements(path, language=language)


def tax_notice(path, language):
    return TaxNotice(path, language=language)


def identity_document(path, language):
    return IdentityDocument(path, language=language)


def get_image(path, doc_type, language):
    switcher = {
        0: account_statements,
        1: tax_notice,
        2: identity_document
    }
    image_class = switcher.get(doc_type)
    return image_class(path, language)


if __name__ == '__main__':
    # Get arguments
    args = parse_args()

    print('*******************************************')
    print('TESSERACT VERSION : {}'.format(pytesseract.get_tesseract_version()))
    print('*******************************************')

    if args.path is None:
        print('Error: You must give a path to a document. Use python ocr_cezam.py -h for more information.')

    set_locale(args.lang)

    # Get image class
    image = get_image(args.path, args.type, args.lang)

    # Process image (create folder, separate pdf pages to different images, process images)
    image.processing()
    
    # Extract image information
    print('PDF path : {}'.format(image.file_path))
    print('Processed path : {}'.format(image.processed_file_path))
    #? Pas sur que l'extract text à cet endroit soit pertinent
    # image.extract_text(save_file=True)
    image.parse_fields()
    print('Bank info :')
    print('Name : {}'.format(image.bank_name))
    print('Address : {}'.format(image.agency_address))
    print('Phone : {}'.format(image.agency_phone))
    print('Email : {}'.format(image.agency_email))

    print('Client info :')
    print('First name : {}'.format(image.first_name))
    print('Last name : {}'.format(image.last_name))
    print('Address : {}'.format(image.address))
    print('Month : {}'.format(image.statement_month))
    print('Year : {}'.format(image.statement_year))

    print('Relevé :')
    print(image.statement_tables)
    # print('TEXT :\n {}'.format(image.file_text_data['text']))
    # print('TEXT :\n {}'.format(image.file_text_data))
