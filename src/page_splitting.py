import re
import os
import time
from typing import List

import cv2
import numpy as np
import matplotlib.pylab as plt
import pytesseract
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import imutils

import functions as f


def running_mean(x, n=3):
    return np.convolve(x, np.ones((n,)) / n, mode='valid')


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def find_local_minima_and_maximas_indexs(data):
    """ From https://stackoverflow.com/a/9667121"""
    minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
    maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max
    return minima, maxima


def combine_points_with_smallest_distance(points: List, merge_threshold):
    points = list(points)
    new_points = []
    if len(points) > 1:
        diff = np.diff(points)
        index = np.argmin(diff)
        if diff[index] < merge_threshold:
            if index > 0:
                new_points += points[:index]
            new_points += [(points[index] + points[index + 1]) / 2.0]
            if index < len(points) - 1:
                new_points += points[index + 2:]
        else:
            new_points = points
    else:
        new_points = points
    return new_points


def combine_lines_within_threshold(lines: List[int], merge_threshold: int = 10):
    """ Combine any lines that are within the threshold and replace them by their average, works recursively, starting
    with the lines closest together.
    """
    new_lines = lines
    if len(lines) > 0:
        merging = True
        while merging:
            nl = combine_points_with_smallest_distance(new_lines, merge_threshold)
            if len(nl) < len(new_lines):
                new_lines = nl
                merging = True
            else:
                merging = False

    return [int(round(x)) for x in new_lines]


def get_average_value(gray, axis, step=5, width=None):
    """ Average pixel values either along the x or y direction to get an average blackness.
    gray:  image, numpy array
    axis: either 'x'/1 or 'y'/0
    step: how many pixels to move on after averaging a line
    width: how many lines of pixels to average together. Set equal to the step by default.
    """
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
    while top_index < gray.shape[ax] - width:
        if ax == 0:
            hl = gray[top_index:top_index + width, :]
        if ax == 1:
            hl = gray[:, top_index:top_index + width]

        mean = np.average(hl[:, :])
        pos.append(top_index)
        value.append(mean)
        top_index += step

    return pos, value


def find_horizontal_lines(gray):
    """ Find main outer horizontal lines """
    margin = int(gray.shape[0]/5)
    pos, value = get_average_value(gray[:margin, :], axis='y', step=5)
    top_line_y_pos = pos[value.index(min(value))]

    new_start = top_line_y_pos + int(margin/4)
    pos, value = get_average_value(gray[new_start:margin, :], axis='y', step=5)
    top_2nd_line_y_pos = pos[value.index(min(value))] + new_start

    start = gray.shape[0] - margin
    pos, value = get_average_value(gray[start:, :], axis='y', step=5)
    bottom_line_y_pos = pos[value.index(min(value))] + start

    return top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos


def find_vertical_lines(gray):
    """ Find main outer vertical lines """
    margin = int(gray.shape[1] / 3)
    left_offset = int(gray.shape[1] / 20)
    pos, value = get_average_value(gray[:, left_offset:margin], axis='x', step=5)
    left_x_pos = pos[value.index(min(value))] + left_offset

    start = gray.shape[1] - margin
    pos, value = get_average_value(gray[:, start:], axis='x', step=5)
    right_x_pos = pos[value.index(min(value))] + start

    return left_x_pos, right_x_pos


def find_outer_boxing_lines_of_table(gray, save_path=None):
    top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos = find_horizontal_lines(gray)

    left_x_pos, right_x_pos = find_vertical_lines(gray)

    if save_path:
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

        cv2.imwrite(save_path, gray)

    return (top_line_y_pos, top_2nd_line_y_pos, bottom_line_y_pos), (left_x_pos, right_x_pos)


def get_smooth_ave_value_and_minima(image, step=1, width=8, threshold=200):
    pos, value = get_average_value(image, axis='y', step=step, width=width)
    pos, value = pos[3:-3], running_mean(value, 7)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    minima_indexes = np.array([x for x in minima if value[x] < threshold])
    minima_indexes = np.array(combine_lines_within_threshold(minima_indexes, 6))
    minimas_pos = pos[minima_indexes]
    return pos, value, minima_indexes, minimas_pos


def rotate_image(gray, cuttoff=30):
    # Check if we need to rotate the page
    ys, xs = find_outer_boxing_lines_of_table(gray)

    line = gray[ys[2] - 30:ys[2] + 50, :]
    margin = 1000

    pos, value, minima_indexes, minimas_pos_left = get_smooth_ave_value_and_minima(line[:, 0:margin].copy())
    # plt.figure(figsize=(10, 6))
    # plt.plot(pos, value, '-r', lw=1)
    # plt.plot(pos[minima_indexes], value[minima_indexes], 'oc', ms=2)

    pos, value, minima_indexes, minimas_pos_right = get_smooth_ave_value_and_minima(line[:, -1 * margin:].copy())
    # plt.plot(pos, value, '-b', lw=1)
    # plt.plot(pos[minima_indexes], value[minima_indexes], 'sk', ms=2)
    # plt.show()

    if len(minimas_pos_left) == len(minimas_pos_right) == 1:
        yl, yr = minimas_pos_left[0], minimas_pos_right[0]
        distance = gray.shape[1] - margin
        theta = np.arcsin((yl - yr) / distance)
        print("theta = {} ({})".format(theta, theta * (180 / np.pi)))
        theta = theta * (180 / np.pi)
        if 2 > np.abs(theta) > 0.1:
            print("Rotating")
            gray = imutils.rotate_bound(gray, theta)
            gray = gray[cuttoff:-cuttoff, cuttoff:-cuttoff]

    return gray


def find_minimas(im, axis='x', min=0, max=150, step=2, width=5, merge_threshold=5, mean=None):
    pos, value = get_average_value(im, axis=axis, step=step, width=width)

    pos, value = np.array(pos), np.array(value)

    if mean is not None:
        assert mean % 2 == 1
        st = int((mean - 1) / 2)
        pos, value = pos[st:-st], running_mean(value, mean)

    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if min < value[x] < max])
    lines = pos[sel_minima]
    lines = combine_lines_within_threshold(lines, merge_threshold)
    return lines


def trim_to_table(gray, threshold=150):
    """ Trim image to the part that is the table. """
    ys, xs = find_outer_boxing_lines_of_table(gray)
    table = gray[ys[1]:ys[2], xs[0]:xs[1]]

    for vt in [threshold, threshold+10, threshold+20, threshold+30, threshold+40, threshold+50, threshold+60]:
        try:
            vertical_lines = find_minimas(table.copy(), axis='x', max=vt, step=2, width=6, merge_threshold=12)
        except IndexError:
            vertical_lines = []
        if len(vertical_lines) > 4:
            break
    vertical_lines = np.array(vertical_lines) + 2  # Center them on the line
    horizontal_lines = find_minimas(table.copy(), axis='y', max=threshold, step=2, width=5, merge_threshold=5)
    text_lines = find_minimas(table.copy(), axis='y', min=200, max=240, step=2, width=5, merge_threshold=6, mean=7)

    return vertical_lines, horizontal_lines, text_lines, table


def plot_table_with_lines(vertical_lines, horizontal_lines, txt_lines, table, save_path):
    table = cv2.cvtColor(table, cv2.COLOR_GRAY2RGB)
    for vl in vertical_lines:
        x1, y1, x2, y2 = vl, 0, vl, table.shape[0]
        cv2.line(table, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for hl in horizontal_lines:
        x1, y1, x2, y2 = 0, hl, table.shape[1], hl
        cv2.line(table, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for hl in txt_lines:
        x1, y1, x2, y2 = 0, hl, table.shape[1], hl
        cv2.line(table, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imwrite(save_path, table)


def find_edges(gray, OUT):
    top_index = 0
    step = 2
    pos, value = [], []
    while top_index < gray.shape[0] - step:
        hl = gray[top_index:top_index + step, :]
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


def add_border(im, border=40):
    """ Add border around image """
    value = (255, 255, 255)
    top, bottom, left, right = border, border, border, border
    borderType = cv2.BORDER_CONSTANT
    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType, None, value)
    return dst


def segment_image(gray):
    """ Find table and split it into individual lines. """
    vertical_lines, horizontal_lines, txt_lines, table = trim_to_table(gray)

    # TODO: deal with cases where there are headings in the table
    # api = PyTessBaseAPI(lang='ita', psm=PSM.SINGLE_WORD) #PSM.SINGLE_BLOCK)

    text = []
    plot_table_with_lines(vertical_lines, [], [], table, '/home/edgar/OCR/out2/main_vertical_lines.png')
    if len(vertical_lines) != 5:
        print('Could not find main 5 dividing vertical lines, returning nothing')
        return text

    half_median_height = int(np.median(np.diff(txt_lines)) / 2)
    start_time = time.time()
    for ii, line_y in enumerate(txt_lines):
        if ii > 0:
            print("{} / {} after {} seconds ({} for the last line)".format(ii + 1, len(txt_lines),
                                                                           time.time() - start_time,
                                                                           time.time() - last_time))
        last_time = time.time()

        top, bottom = line_y - half_median_height, line_y + half_median_height
        line_im = table[top:bottom, :]
        plot_table_with_lines(vertical_lines, [], [], line_im, '/home/edgar/OCR/out2/line_{}-{}.png'.format(top, bottom))
        line_text = ocr_table_line(line_im, vertical_lines)
        text.append(line_text)
    return text


def ocr_table_line(line_im, back_up_main):
    columns = {0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 5}
    line_text = []
    for vt in [20, 30, 40, 50, 80]:
        try:
            main = find_vertical_lines_in_line(line_im, threshold=vt, width=5, merge_threshold=14, cutoff=30)
        except Exception:
            main = []
        if len(main) > 4:
            break
    if len(main) != 5:
        print("Found {} main columns, not {}, falling back on backup".format(len(main), 5))
        # Use backup mains
        main = back_up_main
    # Take each row in turn
    previous_x = 0
    parition_lines = list(main) + [line_im.shape[1]]
    for ii, ml in enumerate(sorted(parition_lines)):
        if (ml - previous_x) > 40:
            xs, xe = previous_x + 10, ml - 4
            part_im = line_im[:, xs:xe].copy()

            if ii != 0:
                for vt in [180, 200, 210, 220]:
                    minor = find_vertical_lines_in_line(part_im, threshold=vt, width=2, merge_threshold=10, cutoff=2)

                    if len(minor) > columns[ii] - 1:
                        break
            else:
                minor = []

            lines = list(minor) + [part_im.shape[1]]
            texts = split_image_by_lines(part_im, lines)

            # Make sure we have the right number of columns
            if len(texts) != columns[ii]:
                if len(texts) < columns[ii] and len(texts[0].split('|')) == columns[ii]:
                    texts = texts[0].split('|')

                while len(texts) < columns[ii]:
                    texts.append('')

                texts = texts[:columns[ii]]
            if len(texts) != columns[ii]:
                print("len(texts), columns[{}] = {}, {} ".format(ii, len(texts), columns[ii]))
            line_text += texts
            previous_x = ml

    return line_text


def split_image_by_lines(image, lines):
    px = 0
    texts = []
    for jj, mil in enumerate(sorted(lines)):
        if (mil - px) > 40:
            xis, xie = px + 3, mil - 1
            cell_im = image[:, xis:xie].copy()

            # api.SetImage(Image.fromarray(part_im))
            # string = api.GetUTF8Text()
            string = pytesseract.image_to_string(cell_im, lang='ita', config='--psm 7')
            texts.append(string.replace(',', '').replace('\n', '').replace('\f', ''))
            px = mil
    return texts


def find_vertical_lines_by_looking_at_edge(line_im, threshold=210, edge=8, width=2, step=2):
    """ Look at top edge to find vertical lines in image. """
    pos, value = get_average_value(line_im[:edge, :].copy(), axis='x', step=step, width=width)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])
    vertical_lines = []
    if len(sel_minima) > 0:
        vertical_lines = pos[sel_minima]

    # plt.figure()
    # plt.plot(pos, value, 'o-r', lw=2, ms=3)
    # plt.plot(pos[sel_minima], value[sel_minima], 'db', ms=4)
    # plt.show()
    return vertical_lines


def find_vertical_lines_in_line(line_im, threshold=40, width=5, merge_threshold=10, cutoff=30):
    edge_width = int(line_im.shape[0] / 4.)
    verticals = find_vertical_lines_by_looking_at_edge(line_im[:, cutoff:-cutoff].copy(), threshold=threshold,
                                                       edge=edge_width, width=width)
    verticals = combine_lines_within_threshold(verticals, merge_threshold=merge_threshold)

    verticals = np.array(verticals)
    verticals = verticals + cutoff
    return verticals


def find_all_vertical_lines(line_im):
    """ Find all vertical lines in an image of a single line of the table.
    Expect 5 major lines and 12-14 minor lines.
    """
    cutoff, small_cutoff = 30, 15
    edge_width = int(line_im.shape[0] / 4.)
    main_verticals = find_vertical_lines_by_looking_at_edge(line_im[:, cutoff:-cutoff].copy(), threshold=40,
                                                            edge=edge_width, width=5)
    if len(main_verticals) < 5:
        main_verticals = find_vertical_lines_by_looking_at_edge(line_im[:, cutoff:-cutoff].copy(), threshold=125,
                                                                edge=edge_width, width=5)
    main_verticals = combine_lines_within_threshold(main_verticals, merge_threshold=10)

    main_verticals = np.array(main_verticals)
    main_verticals = main_verticals + cutoff

    # Now find the lines inbetween
    minor_lines = []
    previous_x = 0
    parition_lines = list(main_verticals) + [line_im.shape[1]]
    for ml in sorted(parition_lines):
        if (ml - previous_x) > 2 * small_cutoff:
            xs, xe = previous_x + small_cutoff, ml - 4
            part_im = line_im[:, xs:xe].copy()

            lines = find_vertical_lines_by_looking_at_edge(part_im, threshold=210, edge=edge_width, width=2)
            minor_lines += [x + xs for x in lines]

            previous_x = ml
    minor_lines = combine_lines_within_threshold(minor_lines, merge_threshold=10)

    return main_verticals, minor_lines
