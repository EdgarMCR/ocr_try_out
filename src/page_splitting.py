import re
import os
import time
from typing import List

import cv2
import numpy as np
import matplotlib.pylab as plt
import pytesseract

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
    """ Find main outer vertical lines """
    pos, value = get_average_value(gray[:, 0:750], axis='x', step=5)
    left_x_pos = pos[value.index(min(value))]

    start = gray.shape[1] - 750
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


def trim_to_table(gray):
    """ Trim image to the part that is the table. """
    ys, xs = find_outer_boxing_lines_of_table(gray)
    table = gray[ys[1]:ys[2], xs[0]:xs[1]]

    threshold = 150

    pos, value = get_average_value(table.copy(), axis='x', step=2, width=5)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])
    vertical_lines = pos[sel_minima]

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
    return vertical_lines, horizontal_lines, txt_lines, table


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


def find_edges(gray):
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
    """ Add border around image """
    value = (255, 255, 255)
    top, bottom, left, right = border, border, border, border
    borderType = cv2.BORDER_CONSTANT
    dst = cv2.copyMakeBorder(im, top, bottom, left, right, borderType, None, value)
    return dst


def split_line_into_rows(line_im, vertical_lines):
    texts = []

    for ii in range(1, len(vertical_lines)):
        xs, xe = vertical_lines[ii - 1], vertical_lines[ii]
        if xe - xs > 10:
            part = line_im[:, xs + 5:xe - 5]
            text = f.get_ocr_as_text_output(part, info=True)
            if ii > 1:
                match = re.search(r'[a-zA-Z]{4,25}|\d{1,5}|>>|"', text.replace(',', ''))
                if match:
                    texts.append(match.group())

    return ', '.join(texts)


def segment_image(gray):
    """ Find table and split it into individual lines. """
    vertical_lines, horizontal_lines, txt_lines, table = trim_to_table(gray)

    # TODO: deal with cases where there are headings in the table

    half_median_height = int(np.median(np.diff(txt_lines))/2)
    text = []
    start_time = time.time()
    for ii, line_y in enumerate(txt_lines):
        if ii > 0:
            print("{} / {} after {} seconds ({} for the last line)".format(ii+1, len(txt_lines), time.time() - start_time, time.time() - last_time))
        last_time = time.time()
        line_text = []
        top, bottom = line_y - half_median_height, line_y + half_median_height
        line_im = table[top:bottom, :]
        main, minor = find_all_vertical_lines(line_im)
        if len(main) == 5:
            #Take each row in turn
            previous_x = 0
            parition_lines = list(main) + list(minor) + [line_im.shape[1]]
            for ml in sorted(parition_lines):
                if (ml - previous_x) > 40:
                    xs, xe = previous_x + 10, ml - 4
                    part_im = line_im[:, xs:xe].copy()
                    string = pytesseract.image_to_string(part_im, lang='ita', config='--psm 7')
                    match = re.search(r'[a-zA-Z]{4,25}|\d{1,5}|>>|"', string.replace(',', ''))
                    if match:
                        line_text.append(match.group())
                    previous_x = ml
        text.append(line_text)
    return text


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


def find_vertical_lines_by_looking_at_edge(line_im, threshold=210, edge=8, width=2, step=2):
    """ Look at top edge to find vertical lines in image. """
    pos, value = get_average_value(line_im[:edge, :].copy(), axis='x', step=step, width=width)

    pos, value = np.array(pos), np.array(value)
    minima, _ = find_local_minima_and_maximas_indexs(value)
    sel_minima = np.array([x for x in minima if value[x] < threshold])
    vertical_lines = []
    if len(sel_minima) > 0:
        vertical_lines = pos[sel_minima]

    return vertical_lines


def find_all_vertical_lines(line_im):
    """ Find all vertical lines in an image of a single line of the table.
    Expect 5 major lines and 12-14 minor lines.
    """
    cutoff, small_cutoff = 30, 15
    edge_width = int(line_im.shape[0] / 4.)
    main_verticals = find_vertical_lines_by_looking_at_edge(line_im[:, cutoff:-cutoff].copy(), threshold=40,
                                                            edge=edge_width, width=5)
    main_verticals = combine_lines_within_threshold(main_verticals, merge_threshold=10)

    main_verticals = np.array(main_verticals)
    main_verticals = main_verticals + cutoff

    # No find the lines inbetween
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




