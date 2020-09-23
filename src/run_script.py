import re
import os
import time
from typing import List

import cv2
import numpy as np
import matplotlib.pylab as plt

import functions as f
import page_splitting as ps

OUT = '/home/edgar/OCR/out/'


def helper_function_segment_line():
    f = '/home/edgar/PycharmProjects/OCR/src/test/line_test_data/line_titel_ex_1.png'
    line_im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    # vertical_lines = find_vertical_lines_by_looking_at_edge(line_im)
    vertical_lines, minor_lines = ps.find_all_vertical_lines(line_im)
    # line_txt = split_line_into_rows(line_im.copy(), vertical_lines)
    # print(line_txt)

    line_im = cv2.cvtColor(line_im, cv2.COLOR_GRAY2RGB)
    for vl in vertical_lines:
        x1, y1, x2, y2 = vl, 0, vl, line_im.shape[0]
        cv2.line(line_im, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for vl in minor_lines:
        x1, y1, x2, y2 = vl, 0, vl, line_im.shape[0]
        cv2.line(line_im, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imwrite(OUT + 'line_title_ex_wut_lines.png', line_im)


def main():
    """
    1. Find longest horizontal and vertical line
    2. Find position of horizontal text lines
    3. Dissect
    4. OCR
    5. Reassemble
    """
    path = '/home/edgar/OCR/Scuole_Primarie_1863_page12.png'
    out = '/home/edgar/OCR/out/'
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imwrite(out + 'edges.jpg', edges)

    # find_edges(gray)
    # find_lines(gray)
    # vertical_lines, horizontal_lines, txt_lines, table = ps.trim_to_table(gray)
    # ps.plot_table_with_lines(vertical_lines, horizontal_lines, txt_lines, table, OUT + 'table_with_lines.png')

    # outf = OUT + 'lines/'
    # if not os.path.exists(outf): os.mkdir(outf)

    text = ps.segment_image(gray)

    with open(OUT + 'Scuole_Primarie_1863_page12.csv', 'w') as f:
        for line in text:
            f.write('{}\n'.format(','.join(line)))

    # helper_function_segment_line()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(" --- Ran in {:.3f} seconds --- ".format(time.time() - start_time))
