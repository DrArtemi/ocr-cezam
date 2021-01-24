import os
import re
import cv2
import pytesseract

import numpy as np
import pandas as pd

from pytesseract.pytesseract import Output
from utils.utils import flatten, process_text


def sort_contours(cnts, method="left-to-right"):
    """This function sort the countours wrt argument method.

    Args:
        cnts (list): List of contours
        method (str, optional): Sorting method. Defaults to "left-to-right".

    Returns:
        tuple(list, list): Sorted contours list and respective bounding boxes.
    """
    reverse = False
    i = 0    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True    
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def remove_overlapping_tables(tables):
    """This function remove tables that are overlapping.
    Overlapping tables are false positive tables.

    Args:
        tables (list): Tables list.

    Returns:
        list: Cleaned tables list.
    """
    tables_test = tables.copy()
    tables_test.sort(key=lambda t: t[2] + t[3], reverse=True)
    new_tables = []
    for table2 in tables_test:
        if len(new_tables) == 0:
            new_tables.append(table2)
        else:
            check = True
            for table1 in new_tables:
                if (table2[0] + table2[2] <= table1[0] + table1[2] and\
                    table2[0] >= table1[0] and\
                    table2[1] >= table1[1] and\
                    table2[1] + table2[3] <= table1[1] + table1[3]):
                   check = False 
            if check:
                new_tables.append(table2)
    return new_tables
    

def store_boxes_to_tables(tables, boxes):
    """This function store every box in it's table.

    Args:
        tables (list): List of detected tables.
        boxes (list): List of detected boxes.

    Returns:
        list: List of boxes, arranged by table.
    """
    table_boxes = []
    for table in tables:
        tx, ty, tw, th = table
        table_boxes.append([])
        for b in boxes:
            x, y, w, h = b
            # If the box coords are IN the table coords,
            # assign box to table
            if tx < x < tx + tw and tx < x + w < tx + tw and\
                ty < y < ty + th and ty < y + h < ty + th:
                table_boxes[-1].append(b)
        table_boxes[-1].sort(key = lambda x: x[1])
    return table_boxes


def tables_to_row_col_tables(tables):
    tables_r_c = []
    previous = None
    row_thresh = 15
    for table in tables:
        row = []
        tables_r_c.append([])
        for j, b in enumerate(table):
            if j == 0:
                row.append(b)
                previous = b
            else:
                if previous[1] - row_thresh <= b[1] <= previous[1] + row_thresh:
                    row.append(b)
                    previous = b
                    if j == len(table) - 1:
                        tables_r_c[-1].append(row)
                else:
                    tables_r_c[-1].append(row)
                    row = []
                    previous = b
                    row.append(b)
    return tables_r_c


def get_tables_columns_info(tables):
    tables_max_col = [0] * len(tables)
    tables_centers = [None] * len(tables)
    for i, table in enumerate(tables):
        for row in table:
            n_col = len(row)
            if n_col > tables_max_col[i]:
                tables_max_col[i] = n_col
                # Retrieving the center of each column
                center = [int(col[0] + col[2] / 2) for col in row]
                tables_centers[i] = np.array(center)
                tables_centers[i].sort()
    return tables_max_col, tables_centers


def arrange_tables(tables, tables_max_col, tables_centers):
    tables_arranged = []
    for i, table in enumerate(tables):
        tables_arranged.append([])
        for j, row in enumerate(table):
            l = [[] for _ in range(tables_max_col[i])]
            for col in row:
                diff = abs(tables_centers[i] - (col[0] + col[2] / 4))
                min_dist = min(diff)
                idx = list(diff).index(min_dist)
                l[idx] = col

            tables_arranged[-1].append(l)
    return tables_arranged


def detect_and_arrange_text(tables, bitnot, arrange_mode, debug_folder):
    debug_cells = None if debug_folder is None else os.path.join(debug_folder, 'cells')
    if debug_cells is not None:
        if not os.path.exists(debug_cells):
            os.makedirs(debug_cells)
    tables_content = []
    tables_bb = []
    tags = ['débit', 'crédit', 'debit', 'credit']
    for i, table in enumerate(tables):
        tables_content.append([])
        tables_bb.append([])
        tagged_columns = []
        for r, row in enumerate(table):
            tables_content[-1].append([])
            tables_bb[-1].append([])
            for c, col in enumerate(row):
                text = [] if arrange_mode == 1 else ' '
                used_bb = []
                if len(col) > 0:
                    y, x, w, h = col
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=1)
                    #* Image cleaning
                    if debug_cells is not None:
                        cv2.imwrite(os.path.join(debug_cells, 'test_{}_{}_bf.jpg'.format(r, c)), erosion)
                    # Close small dots
                    clean_dots = cv2.morphologyEx(src=erosion, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
                    # Resharpen our text by making binary img
                    cleaned = cv2.threshold(clean_dots, 170, 255, cv2.THRESH_BINARY)[1]
                    if debug_cells is not None:
                        cv2.imwrite(os.path.join(debug_cells, 'test_{}_{}_af.jpg'.format(r, c)), cleaned)
                    if arrange_mode == 0:
                        # --psm 4 pour assumer ma cellule comme un seul bloc.
                        text = pytesseract.image_to_string(cleaned, config='--psm 4', lang='fra')
                        if len(text) == 0:
                            text = pytesseract.image_to_string(cleaned, config='--psm 3', lang='fra')
                        text = re.sub('\x0c',  '', text)
                    else:
                        text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, config='--psm 4', lang='fra')
                        text, conf, bb = process_text(text_data)
                        used_bb = []
                        for j in range(len(text)):
                            # Determine if column is credit/debit
                            is_num = True
                            for tag in tags:
                                for word in text[j]:
                                    if tag == word.lower():
                                        is_num = False
                                        tagged_columns.append(c)
                            text[j] = ' '.join(text[j])
                            text[j] = re.sub('\x0c',  '', text[j])
                            # If column is credit/debit and cell is number, treat again cell
                            if c in tagged_columns and is_num:
                                x1, y1, w1, h1 = bb[j][0]
                                x2, y2, w2, h2 = bb[j][-1]
                                cutted_img = cleaned[y1:y1+h1, x1:x2+w2]
                                cutted_img = 255 - cutted_img
                                cutted_border = cv2.copyMakeBorder(cutted_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0])
                                cutted_resizing = cv2.resize(cutted_border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                # cv2.imwrite('table_recognition/test_{}_{}_{}_bf.jpg'.format(r, c, j), cutted_resizing)
                                txt = pytesseract.image_to_string(cutted_resizing, config='--psm 7 -c tessedit_char_whitelist=0123456789', lang='eng')
                                txt = txt.replace(' ', '').replace('\n', '').replace('\x0c', '')
                                # txt = re.sub('\n\x0c',  '', txt)
                                if len(txt) > 2:
                                    txt = txt[:-2] + ',' + txt[-2:]
                                text[j] = txt
                            used_bb.append(bb[j][0][1])
                    
                tables_content[-1][-1].append(text) 
                tables_bb[-1][-1].append(used_bb)
    return tables_content, tables_bb


def prepare_tables_for_df(tables, tables_bb):
    prepared_tables = []
    for t in range(len(tables)):
        threshold = 15
        nb_cols = len(tables[t][0])
        final_tab = []
        for i, row in enumerate(tables[t]):
            #* On récupère les rows différentes à partire des bb
            # Flatten our bounding boxes
            flattened = flatten(tables_bb[t][i])
            # Remove close values
            res = []
            for item in sorted(flattened):
                if len(res) == 0 or item > res[-1] + threshold:
                    res.append(item)
            nb_rows = len(res)
            final_tab.append([['' for _ in range(nb_cols)] for _ in range(nb_rows)])
            for j, col in enumerate(row):
                #* On regarde la col j fu final tab
                for k, txt in enumerate(col):
                    for b in range(len(res)):
                        if res[b] - threshold <= tables_bb[t][i][j][k] <= res[b] + threshold:
                            final_tab[-1][b][j] = txt

        prepared_tables.append([row for row_cell in final_tab for row in row_cell])
    return prepared_tables


def draw_table_detection_bb(img, tables_boxes, tables, debug_folder):
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i, table in enumerate(tables_boxes):
        tx, ty, tw, th = tables[i]
        image = cv2.rectangle(image, (tx,ty), (tx+tw,ty+th), (255, 0, 0), 2)
        for box in table:
            x, y, w, h = box
            image = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_folder, "table_detection.jpg"), image)
    

def process_tables(file_path, debug_folder=None, arrange_mode=0, semiopen_table=False):
    if debug_folder is not None:
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
    # Load image as 1D array
    img = cv2.imread(file_path, 0)
    if img is None:
        print('Process table : error while trying to load {}'.format(file_path))
        return []

    # thresholding the image to a binary image (only 0 or 255)
    _, img_bin = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    
    # Invert image (text become white, background black)
    img_bin = 255 - img_bin
    
    # If debug folder is not None, draw inverted image
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, 'inverted.png'), img_bin)
    
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // 100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    
    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    
    # If debug folder is not None, draw vertical lines image
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, "vertical.jpg"), vertical_lines)
    # If debug folder is not None, draw horizontal lines image
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, "horizontal.jpg"), horizontal_lines)
    
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # This part is used in case of open table
    if semiopen_table:
        # Detect contours for following box detection
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
        
        min_x, min_y = img.shape[1], img.shape[0]
        max_x = max_y = 0
        
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            if x > 0 and y > 0 and x+w < img.shape[1] and y+h < img.shape[0]:
                min_x, max_x = min(x, min_x), max(x+w, max_x)
                min_y, max_y = min(y, min_y), max(y+h, max_y)
        if max_x - min_x > 0 and max_y - min_y > 0:
            cv2.rectangle(img_vh, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    # If debug folder is not None, draw horizontal and vertical lines merges in 1 image
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, "img_vh.jpg"), img_vh)
    
    # Overlapping our table lines on our original image
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    # If debug folder is not None, draw overlapped image
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, "bitnot.jpg"), bitnot)
    
    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    
    # Create list box to store all boxes in
    boxes = []
    # Get position (x,y), width and height for every contour and show the contour on image
    tables = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #! Not genric at ALL, would be better to find another method !
        # A cell can't be more than 80% of the image
        if w < img.shape[1] * 0.8:
            boxes.append([x, y, w, h])
        # If it is, we process it as a table
        elif w < img.shape[1] or h < img.shape[0]:
            tables.append([x, y, w, h])

    # Remove overlaping tables
    tables = remove_overlapping_tables(tables)
    # Remove overlaping boxes
    boxes = remove_overlapping_tables(boxes)
    # Store boxes into their respective table
    tables_boxes = store_boxes_to_tables(tables, boxes)
        
    # Remove empty tables
    idx_to_rm = []
    for i in range(len(tables)):
        if len(tables_boxes[i]) == 0:
            idx_to_rm.append(i)
    tables = [t for i, t in enumerate(tables) if i not in idx_to_rm]
    tables_boxes = [t for i, t in enumerate(tables_boxes) if i not in idx_to_rm]
    
    # If debug folder is not None, draw table detection
    if debug_folder is not None:
        draw_table_detection_bb(img, tables_boxes, tables, debug_folder)
    
    # For each table, create a list of rows in a list of columns
    tables_r_c = tables_to_row_col_tables(tables_boxes)
    
    # calculating maximum number of cells, and get center of each column
    tables_max_col, tables_centers = get_tables_columns_info(tables_r_c)
    
    # Regarding the distance to the columns center, the boxes are arranged in respective order
    tables_arranged = arrange_tables(tables_r_c, tables_max_col, tables_centers)
    
    # Detect text and arrange it and its bounding boxes to make it easy to process
    tables_content, tables_bb = detect_and_arrange_text(tables_arranged,
                                                        bitnot,
                                                        arrange_mode,
                                                        debug_folder)
    
    # If mode is 1, we prepare our tables to create dataframes
    if arrange_mode == 1:
        tables_content = prepare_tables_for_df(tables_content, tables_bb)
    
    # Creating a dataframe of the generated OCR list
    tables_dataframe = []
    for table_content in tables_content:
        # We create a dataframe for each table with the first row being the columns names
        if len(table_content) > 1:
            columns = table_content[0]
            for i in range(len(columns)):
                columns[i] = '{} - {}'.format(i, columns[i])
            tables_dataframe.append(pd.DataFrame(np.array(table_content[1:]), columns=table_content[0]))
        else:
            tables_dataframe.append(pd.DataFrame(np.array(table_content)))
    
    return tables_dataframe
    
    
