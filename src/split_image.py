import re
import os
import time
from typing import List

import cv2
import numpy as np
import matplotlib.pylab as plt

import functions as f

OUT = '/home/edgar/OCR/out/'


def find_lines_trial(gray):
    path = '/home/edgar/OCR/Scuole_Primarie_1863_page12.png'
    out = OUT
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imwrite(out + 'edges.jpg', edges)

    find_edges(gray)

    min_length = min(gray.shape[0], gray.shape[1])
    minLineLength = min_length / 4
    maxLineGap = 50

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=minLineLength, maxLineGap=maxLineGap)
    print("Found {} lines".format(len(lines)))

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    for ii in range(len(lines)):
        x1, y1, x2, y2 = lines[ii][0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite(out + 'edges+houghlines5.jpg', edges)
    cv2.imwrite(out + 'img+houghlines5.jpg', img)


def running_mean(x, n=3):
    return np.convolve(x, np.ones((n,))/n, mode='valid')


def find_local_minima_and_maximas_indexs(data):
    """ From https://stackoverflow.com/a/9667121"""
    minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
    maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max
    return minima, maxima


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def get_average_value(gray, axis, step=5, width=None):
    if axis == 'y':
        ax = 0
    elif axis == 'x':
        ax = 1
    elif isinstance(axis, int):
        ax = axis
    else:
        raise RuntimeError("Axis needs to be valid int or 'x' or 'y'")

    top_index = 0
    if width is None:
        width = step
    pos, value = [], []
    while top_index < gray.shape[ax]-width:
        if ax == 0:
            hl = gray[top_index:top_index+width, :]
        if ax == 1:
            hl = gray[:, top_index:top_index + width]

        mean = np.average(hl[:, :])
        pos.append(top_index)
        value.append(mean)
        top_index += step

    # plt.figure()
    # plt.plot(pos, value, 'o-')

    return pos, value


def find_horizontal_lines(gray):
    pos, value = get_average_value(gray[:750, :], axis='y', step=5)
    top_line_y_pos = pos[value.index(min(value))]

    new_start = top_line_y_pos + 100
    pos, value = get_average_value(gray[new_start:750, :], axis='y', step=5)
    top_2nd_line_y_pos = pos[value.index(min(value))] + new_start

    start = gray.shape[0] - 750
    pos, value = get_average_value(gray[start:, :], axis='y', step=5)
    bottom_line_y_pos = pos[value.index(min(value))] + start

    return top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos


def find_vertical_lines(gray):
    pos, value = get_average_value(gray[:, 0:750], axis='x', step=5)
    left_x_pos = pos[value.index(min(value))]

    start = gray.shape[1] - 750
    pos, value = get_average_value(gray[:, start:], axis='x', step=5)
    right_x_pos = pos[value.index(min(value))] + start

    return left_x_pos, right_x_pos


def find_lines(gray, plot=False):
    top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos = find_horizontal_lines(gray)

    print(top_line_y_pos)
    print(top_2nd_line_y_pos)
    print(bottom_line_y_pos)

    left_x_pos, right_x_pos = find_vertical_lines(gray)
    print(left_x_pos, right_x_pos)

    if plot:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        x1, y1, x2, y2 = 0, top_line_y_pos, gray.shape[1], top_line_y_pos
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)
        x1, y1, x2, y2 = 0, top_2nd_line_y_pos, gray.shape[1], top_2nd_line_y_pos
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)
        x1, y1, x2, y2 = 0, bottom_line_y_pos, gray.shape[1], bottom_line_y_pos
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)

        x1, y1, x2, y2 = left_x_pos, 0, left_x_pos, gray.shape[0]
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)
        x1, y1, x2, y2 = right_x_pos, gray.shape[0], right_x_pos, 0
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.imwrite(OUT + 'horizontl_lines.jpg', gray)

    return (top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos), (left_x_pos, right_x_pos)


def trim_to_table(gray):
    ys, xs = find_lines(gray)
    table = gray[ys[1]:ys[2], xs[0]:xs[1]]

    threshold = 150

    pos, value = get_average_value(table.copy(), axis='x', step=2, width=5)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])
    vertical_lines = pos[sel_minima]

    # plt.figure(figsize=(6,4))
    # plt.plot(pos, value, 'o-r ', ms=2, lw=1, label='x')
    # plt.plot(pos[sel_minima], value[sel_minima], 'dy', ms=3, lw=None, label='x')
    # plt.title('x')
    # plt.savefig(OUT + 'x_table_lines.png', dpi=300)
    # plt.figure(figsize=(6,4))

    pos, value = get_average_value(table.copy(), axis='y', step=2, width=5)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])

    horizontal_lines = pos[sel_minima]

    pos_s, value_s = pos[3:-3], running_mean(value, 7)
    minima, _ = find_local_minima_and_maximas_indexs(value_s)
    text_lines = np.array([x for x in minima if 200 < value_s[x] < 240])
    txt_lines = pos[text_lines]

    # plt.plot(pos, value, 's-b', ms=2, lw=1, label='y')
    # plt.plot(pos_s, value_s, 'h:k', ms=2, lw=1, label='y')
    # plt.plot(pos[sel_minima], value[sel_minima], 'dy', ms=3, lw=None, label='x')
    # plt.plot(pos_s[text_lines], value_s[text_lines], 'dg', ms=3, lw=None, label='x')
    # plt.title('y')
    # # plt.savefig(OUT + 'y_table_lines.png', dpi=300)
    # plt.show()

    # table = cv2.cvtColor(table, cv2.COLOR_GRAY2RGB)
    # for vl in vertical_lines:
    #     x1, y1, x2, y2 = vl, 0, vl, table.shape[0]
    #     cv2.line(table, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #
    # for hl in horizontal_lines:
    #     x1, y1, x2, y2 = 0, hl, table.shape[1], hl
    #     cv2.line(table, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #
    # for hl in txt_lines:
    #     x1, y1, x2, y2 = 0, hl, table.shape[1], hl
    #     cv2.line(table, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #
    # cv2.imwrite(OUT + 'table_lines.jpg', table)
    return vertical_lines, horizontal_lines, txt_lines, table


def find_edges(gray):

    top_index = 0
    step = 2
    pos, value = [], []
    while top_index < gray.shape[0]-step:
        hl = gray[top_index:top_index+step, :]
        mean = np.average(hl[:, :])
        print("{} -> {}".format(top_index, mean))
        pos.append(top_index)
        value.append(mean)
        top_index += step

    plt.figure(figsize=(6, 4))
    plt.plot(pos, value, 'o-r', ms=3, lw=2, label='Raw data')
    plt.plot(pos[1:-1], running_mean(value, 3), 's:b', ms=2, lw=1, label='Smoothed Average')
    plt.legend()
    plt.title('Step = {}'.format(step))
    plt.ylabel('Average pixel value [0 is black]')
    plt.xlabel('Y-position')
    plt.savefig(OUT + 'y_mean_pixel_value_step{}.png'.format(step), dpi=300)
    plt.show()


def segment_into_sub_lines(im, segment):
    # find lines in input

    if not segment:
        threshold = 150
        pos, value = get_average_value(im.copy(), axis='x', step=2, width=2)
        pos, value = np.array(pos), np.array(value)
        minima, _ = find_local_minima_and_maximas_indexs(value)
        sel_minima = np.array([x for x in minima if value[x] < threshold])
        vertical_lines = pos[sel_minima]

        # plt.figure(figsize=(6,4))
        # plt.plot(pos, value, 'o-r', ms=2, lw=1, label='x')
        # plt.plot(pos[sel_minima], value[sel_minima], 'dy', ms=3, lw=None, label='x')
        # plt.title('x')
        # plt.show()

        previous_x = None
        texts = []
        for x_line in vertical_lines:
            if previous_x is None:
                if x_line < 40:
                    previous_x = 40
                else:
                    previous_x = x_line
                continue
            part = im[:, previous_x:x_line - 5]
            part = add_border(part)
            # cv2.imshow('min_part {}:{}'.format(previous_x, x_line), part)
            # cv2.waitKey(2)
            # plt.figure()
            # plt.imshow(part, cmap='gray', vmin=0, vmax=255)
            # plt.show()
            text = f.get_ocr_as_text_output(part, info=True)
            texts.append(text)
            previous_x = x_line + 10

        part = im[:, previous_x:-5]
        text = f.get_ocr_as_text_output(part, info=True)
        texts.append(text)

        text = ' '.join(texts)
    else:
        text = f.get_ocr_as_text_output(im)
    return text.replace('\n', '')


def add_border(im, border=40):
    value = (255, 255, 255)
    top, bottom, left, right = border, border, border, border
    borderType = cv2.BORDER_CONSTANT
    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType, None, value)
    return dst


def split_line_into_rows(line_im, vertical_lines):
    texts = []

    for ii in range(1, len(vertical_lines)):
        xs, xe = vertical_lines[ii-1], vertical_lines[ii]
        if xe-xs > 10:
            part = line_im[:, xs+5:xe-5]
            cv2.imwrite(OUT + 'line_segmented_{}-{}.png'.format(xs, xe), part)
            text = f.get_ocr_as_text_output(part, info=True)
            if ii > 1:
                match = re.search(r'[a-zA-Z]{4,25}|\d{1,5}|>>|"', text.replace(',', ''))
                if match:
                    texts.append(match.group())

    return ', '.join(texts)


def segment_image(gray):
    vertical_lines, horizontal_lines, txt_lines, table = trim_to_table(gray)

    line_spacing = txt_lines[1] - txt_lines[0]
    top = int(txt_lines[0] - line_spacing/2)
    bottom = int(txt_lines[0] + line_spacing / 2)

    line_im = table[top:bottom, :]
    cv2.imwrite(OUT + 'line_ex.png', line_im)

    print(vertical_lines)
    # line_txt = split_line_into_rows(line_im, vertical_lines)
    # print(line_txt)
    # cv2.imshow('line', line_im)
    # cv2.waitKey(2)


def find_vertical_lines_by_looking_at_edge(line_im):
    bottom = line_im.shape[0]
    width = 8  # pixels
    threshold = 210
    pos, value = get_average_value(line_im[:width, :].copy(), axis='x', step=2, width=2)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])
    vertical_lines = pos[sel_minima]

    return vertical_lines


def helper_function_segment_line():
    f = OUT + 'line_ex.png'
    line_im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    vertical_lines = find_vertical_lines_by_looking_at_edge(line_im)

    line_txt = split_line_into_rows(line_im.copy(), vertical_lines)
    print(line_txt)

    line_im = cv2.cvtColor(line_im, cv2.COLOR_GRAY2RGB)
    for vl in vertical_lines:
        x1, y1, x2, y2 = vl, 0, vl, line_im.shape[0]
        cv2.line(line_im, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imwrite(OUT + 'line_ex_wut_lines.png', line_im)


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
    # trim_to_table(gray)
    # segment_image(gray)
    helper_function_segment_line()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(" --- Ran in {:.3f} seconds --- ".format(time.time() - start_time))
