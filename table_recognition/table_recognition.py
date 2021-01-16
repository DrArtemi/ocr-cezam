import re
import cv2
import pytesseract

import numpy as np
import pandas as pd

# from string import maketrans
from pdf2image import convert_from_path
from pytesseract.pytesseract import Output

pdf = "/media/adrien/Shared/Work/Cezam/documents/OCR/pdf_examples/compte joint_BANQUE POP_012017_bis.pdf"
# pdf = "/media/adrien/Shared/Work/Cezam/documents/OCR/Relevés bancaires/JC Perin_Relevé BNP Sept.pdf"

mode = 1

pages = convert_from_path(pdf)

for i, page in enumerate(pages):
    page.save('page_{}.jpg'.format(i), 'JPEG')

# read your file
file=r'page_0.jpg'
img = cv2.imread(file,0)
print(img.shape)

# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#inverting the image 
img_bin = 255 - img_bin
cv2.imwrite('inverted.png', img_bin)

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
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite("vertical.jpg", vertical_lines)

# Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite("horizontal.jpg", horizontal_lines)

# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_vh.jpg", img_vh)

bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
# Plotting the generated image
cv2.imwrite("bitnot.jpg", bitnot)

# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):    # initialize the reverse flag and sort index
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


# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

# Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# Get mean of heights
mean = np.mean(heights)

#Create list box to store all boxes in  
box = []
image = None
# Get position (x,y), width and height for every contour and show the contour on image
tables = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # if (w < 1000 and h < 500):
    if w < img.shape[1] * 0.8:
        # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        box.append([x,y,w,h])
    elif w < img.shape[1] or h < img.shape[0]:
        tables.append([x, y, w, h])

# Remove overlaping tables
tables_test = tables.copy()
tables_test.sort(key=lambda t: t[2] + t[3], reverse=True)
tables = []
for table2 in tables_test:
    if len(tables) == 0:
        tables.append(table2)
    else:
        for table1 in tables:
            if not (table2[0] + table2[2] <= table1[0] + table1[2] and\
                    table2[0] >= table1[0] and\
                    table2[1] >= table1[1] and\
                    table2[1] + table2[3] <= table1[1] + table1[3]):
                tables.append(table2)

# Store boxes into their respective table
table_boxes = []
for table in tables:
    tx, ty, tw, th = table
    table_boxes.append([])
    for b in box:
        x, y, w, h = b
        if tx < x < tx + tw and tx < x + w < tx + tw and\
            ty < y < ty + th and ty < y + h < ty + th:
            table_boxes[-1].append(b)

# Remove empty tables
idx_to_rm = []
for i in range(len(tables)):
    if len(table_boxes[i]) == 0:
        idx_to_rm.append(i)
tables = [t for i, t in enumerate(tables) if i not in idx_to_rm]
table_boxes = [t for i, t in enumerate(table_boxes) if i not in idx_to_rm]
        
image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for i, table in enumerate(table_boxes):
    tx, ty, tw, th = tables[i]
    image = cv2.rectangle(image, (tx,ty), (tx+tw,ty+th), (255, 0, 0), 2)
    for box in table:
        x, y, w, h = box
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

cv2.imwrite("contour.jpg", image)

print('Nombre de tableaux trouvé(s) : {} ({})'.format(len(tables), tables))

# Add 2 lists for each tables (row and column) 
tables_r_c = []
previous = None
row_thresh = 20
for i, table in enumerate(table_boxes):
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

print("Result row/columns table : ", tables_r_c)
for table in tables_r_c:
    print('----- Table -----')
    print('Nb row : {}'.format(len(table)))
    for i, row in enumerate(table):
        print('Row {} nb cols : {}'.format(i, len(row)))

# calculating maximum number of cells
table_max_col = [0] * len(tables)
table_centers = [None] * len(tables)

for i, table in enumerate(tables_r_c):
    for j, row in enumerate(table):
        n_col = len(row)
        if n_col > table_max_col[i]:
            table_max_col[i] = n_col
            # Retrieving the center of each column
            center = [int(col[0] + col[2] / 2) for col in row]
            table_centers[i] = np.array(center)
            table_centers[i].sort()
        
print('Max columns for each tables : {}'.format(table_max_col))
print('Column centers for each tables : {}'.format(table_centers))

# Regarding the distance to the columns center, the boxes are arranged in respective order
tables_arranged = []
for i, table in enumerate(tables_r_c):
    tables_arranged.append([])
    for j, row in enumerate(table):
        l = [[]] * table_max_col[i]
        for col in row:
            diff = abs(table_centers[i] - (col[0] + col[2] / 4))
            min_dist = min(diff)
            idx = list(diff).index(min_dist)
            l[idx] = col

        tables_arranged[-1].append(l)
        
print('Arranged table : {}'.format(tables_arranged))
print('Nb tables : {}'.format(len(tables_arranged)))
print('Nb rows : {}'.format(len(tables_arranged[0])))
for i, row in enumerate(tables_arranged[0]):
    print('Row {} : Nb cols : {}'.format(i, len(row)))


def process_text(text_data):
    word_list = []
    conf_list = []
    bb_list = []
    parse_text = []
    parse_conf = []
    parse_bb = []
    last_word = ''
    for i, word in enumerate(text_data['text']):
        if word != '':
            word_list.append(word)
            conf_list.append(text_data['conf'][i])
            bb_list.append([text_data['left'][i],
                            text_data['top'][i],
                            text_data['width'][i],
                            text_data['height'][i]])
            last_word = word
        if (last_word != '' and word == '') or (i == len(text_data['text']) - 1):
            if len(word_list) > 0:
                parse_text.append(word_list)
                parse_conf.append(conf_list)
                parse_bb.append(bb_list)
            word_list = []
            conf_list = []
            bb_list = []
    return parse_text, parse_conf, parse_bb

tables_content = []
tables_bb = []
for i, table in enumerate(tables_arranged):
    tables_content.append([])
    tables_bb.append([])
    for r, row in enumerate(table):
        tables_content[-1].append([])
        tables_bb[-1].append([])
        for c, col in enumerate(row):
            text = []
            used_bb = []
            if len(col) > 0:
                y, x, w, h = col
                finalimg = bitnot[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255,255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                dilation = cv2.dilate(resizing, kernel,iterations=1)
                erosion = cv2.erode(dilation, kernel,iterations=1)
                # cv2.imwrite('test_{}_{}_bf.jpg'.format(r, c), erosion)
                #* Image cleaning
                # Close small dots
                clean_dots = cv2.morphologyEx(src=erosion, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
                # Resharpen our text by making binary img
                cleaned = cv2.threshold(clean_dots, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                
                # cv2.imwrite('test_{}_{}_af.jpg'.format(r, c), cleaned)
                if mode == 0:
                # --psm 4 pour assumer ma cellule comme un seul bloc.
                    text = pytesseract.image_to_string(cleaned, config='--psm 4', lang='fra')
                    if len(text) == 0:
                        text = pytesseract.image_to_string(cleaned, config='--psm 3', lang='fra')
                else:
                    text_data = pytesseract.image_to_data(cleaned, output_type=Output.DICT, config='--psm 4', lang='fra')
                    text, conf, bb = process_text(text_data)
                    used_bb = []
                    for i in range(len(text)):
                        text[i] = ' '.join(text[i])
                        text[i] = re.sub('\x0c',  '', text[i])
                        used_bb.append(bb[i][0][1])
                
            tables_content[-1][-1].append(text) 
            tables_bb[-1][-1].append(used_bb)


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def process_table(idx, table):
    threshold = 20
    nb_cols = len(table[0])
    final_tab = []
    for i, row in enumerate(table):
        #* On récupère les rows différentes à partire des bb
        # Flatten our bounding boxes
        flattened = flatten(tables_bb[idx][i])
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
                for t in range(len(res)):
                    if res[t] - threshold <= tables_bb[idx][i][j][k] <= res[t] + threshold:
                        final_tab[-1][t][j] = txt

    final_tab = [row for row_cell in final_tab for row in row_cell]
    return final_tab

process_table(0, tables_content[0])
print(tables_content[0][1])

print('Final content :')
for i in range(len(tables_content)):
    if mode == 1:
        tables_content[i] = process_table(i, tables_content[i])
    print('Table {} content :\n{}'.format(i, tables_content[i]))

# Creating a dataframe of the generated OCR list
tables_dataframe = []
for table_content in tables_content:
    tables_dataframe.append(pd.DataFrame(np.array(table_content[1:]), columns=table_content[0]))

print('Dataframes :')
for df in tables_dataframe:
    print(df)

# Characters to delete
# del_chars =  " ".join(chr(i) for i in list(range(32)) + list(range(127, 256)) if i != 10 and i != 13)
# trans = str.maketrans(del_chars, " " * len(del_chars))

# for i, df in enumerate(tables_dataframe):
#     renames = {}
#     for column in df:
#         renames[column] = column.translate(trans)
#         df[column] = df[column].apply(lambda s: s.translate(trans))
#     tables_dataframe[i] = df.rename(columns=renames)

# for i, df in enumerate(tables_dataframe):
#     tables_dataframe[i] = df.applymap(lambda x: x.encode('unicode_escape').
#                                       decode('utf-8') if isinstance(x, str) else x)

# print('Cleaned dataframes :')
# for df in tables_dataframe:
#     print(df)

# Converting it in a excel-file
for i, df in enumerate(tables_dataframe):
    df.to_excel('page_0_tab_{}.xlsx'.format(i))