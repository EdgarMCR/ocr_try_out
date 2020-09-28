import re
import os
import time
from typing import List

import cv2
import numpy as np
import matplotlib.pylab as plt

import functions as f
import page_splitting as ps
import miscellaneous as misc
import run_script as rs

OUT = '/home/edgar/OCR/out2/'
path = '/home/edgar/OCR/Scuole_Primarie_1863_page13.png'
out = '/home/edgar/OCR/output/'

rs.ocr_pdf_in_folder('/home/edgar/OCR/', out)
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = ps.rotate_image(gray)
# # cv2.imwrite(OUT + 'rotated_11.png', gray)
#
# ps.find_outer_boxing_lines_of_table(gray, save_path=OUT + 'outter_lines_13.png')
# vertical_lines, horizontal_lines, txt_lines, table = ps.trim_to_table(gray)
# ps.plot_table_with_lines(vertical_lines, horizontal_lines, txt_lines, table, OUT + 'table_with_lines13.png')
#
#
# text = ps.segment_image(gray)
# path = OUT + 'Scuole_Primarie_1863_page13.csv'
# with open(path, 'w') as f:
#     for line in text:
#         f.write('{}\n'.format(','.join(line)))
#         print(line)
# misc.clean_csv(path)
