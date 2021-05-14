import re
import os
import time
import traceback
import datetime
import logging
from typing import List
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib.pylab as plt

import functions as f
import page_splitting as ps
import miscellaneous as misc

OUT = '/home/edgar/OCR/out/'
HEADER = ["Circondaria e Comuni", "Scuole Totale", "Scuole Maschili", "Scuole Feminili", "Alunni Totale",
          "Alunni Maschili", "Alunni Feminili", "Inseganti Totale", "Inseganti Maschili", "Inseganti Feminili",
          "Proventi Totale", "Proventi dai Governo", "Proventi dalla Provinc", "Proventi dal Comune",
          "Proventi diversi"]


def ocr_pdf_page(path_pdf, save_folder):
    paths = misc.convert_pdf_to_image(path_pdf, dpi=300)

    for path in paths:
        save_path = os.path.join(save_folder, os.path.basename(path.replace('.png', '.csv')))
        print("save_path = {}".format(save_path))
        if not os.path.exists(save_path):
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = ps.rotate_image(gray)

            print('Starting processing image ...')
            text = ps.segment_image(gray)

            with open(save_path, 'w') as f:
                f.write('{}\n'.format(','.join(HEADER)))
                for line in text:
                    f.write('{}\n'.format(','.join(line)))
            misc.clean_csv(save_path)
        else:
            print("{} already exists, returning".format(save_path))


def ocr_pdf_in_folder(folder, save_folder):
    results = []
    for file in os.scandir(folder):
        if '.pdf' in file.name:
            print("Doing {} at {}".format(file.name, str(datetime.datetime.now())))
            match = re.search(r'page(\d{1,3}).pdf', file.name)
            if match:
                # if int(match.group(1)) % 2 != 0:
                #     print('skipping odd pages')
                #     continue

                try:
                    ocr_pdf_page(file.path, save_folder)
                except Exception as e:
                    print("{} didn't work:{}".format(file.name, repr(e)))


def main():
    """
    1. Find longest horizontal and vertical line
    2. Find position of horizontal text lines
    3. Dissect
    4. OCR
    5. Reassemble
    """
    path = '/home/edgar/OCR/Scuole_Primarie_1863_page14.png'
    out = '/home/edgar/OCR/out2/'
    # ocr_pdf_in_folder(folder='/home/edgar/OCR/', save_folder='/home/edgar/OCR/output')
    # ocr_pdf_page(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imwrite(out + 'edges.jpg', edges)

    # find_edges(gray)
    # find_lines(gray)
    gray = ps.rotate_image(gray)
    ps.find_outer_boxing_lines_of_table(gray, out+'boxing_lines.png')
    vertical_lines, horizontal_lines, txt_lines, table = ps.trim_to_table(gray, threshold=150)
    ps.plot_table_with_lines(vertical_lines, horizontal_lines, txt_lines, table, out + 'table_with_lines14_wo.png')
    #

    # outf = OUT + 'lines/'
    # if not os.path.exists(outf): os.mkdir(outf)

    # text = ps.segment_image(gray)
    # path = out + 'Scuole_Primarie_1863_page110.csv'
    # with open(path, 'w') as f:
    #     for line in text:
    #         f.write('{}\n'.format(','.join(line)))
    #         print(line)
    # misc.clean_csv(path)
    # helper_function_segment_line()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(" --- Ran in {:.3f} seconds --- ".format(time.time() - start_time))
