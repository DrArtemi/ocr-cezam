import os
import datetime
import re
import cv2
import pytesseract
from pytesseract.pytesseract import Output

from utils.utils import EMAIL_RGX, get_json_from_file, process_text, remove_background


def get_bank_id(img):
    cleaned = remove_background(img)
    text = pytesseract.image_to_string(cleaned, lang='fra')
    data = get_json_from_file(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'file_configs/bank_configs/banks.json'
        ))
    for bank_id, bank_pat_list in data.items():
        for bank_pattern in bank_pat_list:
            if bank_pattern in text.lower():
                return bank_id
    return None


def get_address(text, dicts):
    address_id = dicts['address']
    for i, row in enumerate(text):
        for word in row:
            if word.lower() in address_id:
                # Address is most of the time written on 2 rows
                return ' '.join(row + text[i + 1] if i + 1 < len(text) else row)
    return 'N/A'


def get_phone(text, dicts):
    phone_id = dicts['phone']
    for row in text:
        for word in row:
            if word.lower() in phone_id:
                return ''.join(row[1:]).replace(':', '')
    return 'N/A'

def get_email(text):
    for row in text:
        for word in row:
            if re.search(EMAIL_RGX, word):
                return word
    return 'N/A'


def get_full_name(text, dicts):
    person_id = dicts['person_id']
    for row in text:
        last_id = -1
        for j, word in enumerate(row):
            if word.lower() in person_id:
                last_id = j
        if last_id != -1:
            return ' '.join(row[last_id+1:])
    return 'N/A'


def get_client_information(img, bank_utils, dicts, debug):
    zone_client = bank_utils['client_info']
    zone_img = img[int(img.shape[0] * zone_client[1][0]):int(img.shape[0] * zone_client[1][1]),
                   int(img.shape[1] * zone_client[0][0]):int(img.shape[1] * zone_client[0][1])]
    cleaned = remove_background(zone_img)
    if debug is not None:
        cv2.imwrite(debug, cleaned)
    text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, lang='fra')
    text, conf, bb = process_text(text_data)
    # Get client's full name
    full_name = get_full_name(text, dicts)
    # Get client's address
    address = get_address(text, dicts)
    return full_name, address


def get_agency_information(img, bank_utils, dicts, debug):
    zone_bank = bank_utils['bank_info']
    zone_img = img[int(img.shape[0] * zone_bank[1][0]):int(img.shape[0] * zone_bank[1][1]),
                   int(img.shape[1] * zone_bank[0][0]):int(img.shape[1] * zone_bank[0][1])]
    cleaned = remove_background(zone_img)
    if debug is not None:
        cv2.imwrite(debug, cleaned)
    text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, lang='fra')
    text, conf, bb = process_text(text_data)
    # Get agency email
    email = get_email(text)
    # Get agency phone
    phone = get_phone(text, dicts)
    # Get agency address
    address = get_address(text, dicts)
    
    return email, phone, address


def get_date(img, bank_utils, debug):
    date_format = bank_utils['date_format']
    zone_date = bank_utils['date_info']
    zone_img = img[int(img.shape[0] * zone_date[1][0]):int(img.shape[0] * zone_date[1][1]),
                   int(img.shape[1] * zone_date[0][0]):int(img.shape[1] * zone_date[0][1])]
    cleaned = remove_background(zone_img)
    if debug is not None:
        cv2.imwrite(debug, cleaned)
    text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, lang='fra')
    text, conf, bb = process_text(text_data)

    for row in text:
        try:
            date = datetime.datetime.strptime(' '.join(row), date_format)
            return date
        except ValueError:
            pass
        for word in row:
            try:
                date = datetime.datetime.strptime(word, date_format)
                return date
            except ValueError:
                pass
    return 'N/A'
