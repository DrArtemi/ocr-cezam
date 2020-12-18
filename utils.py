import os
import PyPDF2
import subprocess

import cv2 as cv


def remove_dot_background(img, kernel=(5, 5)):
    # Global thresholding and invert color
    ret1, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    # Removing noise
    cv_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)
    opening = cv.morphologyEx(th1, cv.MORPH_OPEN, cv_kernel)

    # Re-invert color
    res = 255 - opening

    return res


def pdf_to_tiff(file_path):
    if not os.path.exists(file_path):
        print('Error: {}: file not found.'.format(file_path))
        return None

    file_path_tiff = file_path.split('.')[:-1]
    file_path_tiff.append('tiff')
    file_path_tiff = '.'.join(file_path_tiff)

    if os.path.exists(file_path_tiff):
        os.remove(file_path_tiff)

    subprocess.run(['convert',
                    '-density', '600',
                    file_path,
                    '-background', 'white',
                    '-alpha', 'background',
                    '-alpha', 'off',
                    '-depth', '8',
                    file_path_tiff])
    
    return file_path_tiff


def save_cv_image(img, original_path, extension, del_original=False):
        new_path = original_path.split('.')[:-1]
        new_path.append(extension)
        new_path = '.'.join(new_path)

        if os.path.exists(new_path):
            os.remove(new_path)

        cv.imwrite(new_path, img)

        if del_original:
            os.remove(original_path)
        return new_path


def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):
    paths = []
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(input_pdf_path, "rb") as input_stream:
        input_pdf = PyPDF2.PdfFileReader(input_stream)

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            paths.append(file_name)
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)
    
    return paths


def write_file_from_text(text, filename):
    with open(filename, "w+") as file:
        file.write(text)