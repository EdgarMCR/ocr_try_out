import os
import math
import statistics
from difflib import SequenceMatcher
import unittest

import cv2

import page_splitting as ps

class TestSplittingLineIntoCells(unittest.TestCase):
    test_dir = 'line_test_data/'

    def test_find_vertical_lines_in_line_1(self):
        path = self.test_dir + 'line_example_1.png'
        line_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        main_lines, minor_lines = ps.find_all_vertical_lines(line_im)

        self.assertEqual(5, len(main_lines))
        self.assertEqual(13, len(minor_lines))

    def test_find_vertical_lines_in_line_2(self):
        path = self.test_dir + 'line_example_2.png'
        line_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        main_lines, minor_lines = ps.find_all_vertical_lines(line_im)

        self.assertEqual(5, len(main_lines))
        self.assertEqual(14, len(minor_lines))

    def test_find_vertical_lines_in_line_3(self):
        path = self.test_dir + 'line_example_3.png'
        line_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        main_lines, minor_lines = ps.find_all_vertical_lines(line_im)

        self.assertEqual(5, len(main_lines))
        self.assertEqual(14, len(minor_lines))

    def test_find_vertical_lines_in_line_4(self):
        path = self.test_dir + 'line_example_4.png'
        line_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        main_lines, minor_lines = ps.find_all_vertical_lines(line_im)

        self.assertEqual(5, len(main_lines))
        self.assertEqual(14, len(minor_lines))

    def test_find_vertical_lines_in_title(self):
        path = self.test_dir + 'line_titel_ex_1.png'
        line_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        main_lines, minor_lines = ps.find_all_vertical_lines(line_im)

        self.assertEqual(0, len(main_lines))
