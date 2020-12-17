import argparse

from file_types import IdentityDocument, TaxNotice, AccountStatements


def parse_args():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("-path", type=str, default=None, help="path to document", required=True)
    parser.add_argument("-path-file", type=str, default=None, help="path to file that lists documents")

    parser.add_argument("-type", type=int, default=0,
                        help="type of the document [0 = relevé de compte, 1 = avis d'imposition, 2 = piece d'identite]",
                        required=True)
    parser.add_argument("-lang", type=str, default='fra', help="document language")

    return parser.parse_args()


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

    if args.path is None and args.path_file is None:
        print('Error: You must give a path to a document. Use python ocr_cezam.py -h for more information.')

    # Get image class
    image = get_image(args.path, args.type, args.lang)

    # Process image (create folder, separate pdf pages to different images, process images)
    image.processing()
    
    # Extract image information
    print('PDF path : {}'.format(image.file_path))
    print('Processed path : {}'.format(image.processed_file_path))
    image.extract_text()
    print('TEXT :\n {}'.format(image.file_text))
    # print('TEXT :\n {}'.format(image.file_text_bb))
    # print('TEXT :\n {}'.format(image.file_text_data))
