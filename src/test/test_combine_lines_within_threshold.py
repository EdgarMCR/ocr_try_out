import unittest

import page_splitting as ps


class TestFindAllVerticalLines(unittest.TestCase):

    def test_combine_lines_within_threshold_no_merge(self):
        lines = [1, 50, 100, 500]
        new_lines = ps.combine_lines_within_threshold(lines, 5)
        self.assertCountEqual(lines, new_lines)

    def test_combine_lines_within_threshold_merge_1(self):
        lines = [1, 50, 100, 495, 500]
        expected = [1, 50, 100, 498]
        new_lines = ps.combine_lines_within_threshold(lines, 6)
        self.assertCountEqual(expected, new_lines)

    def test_combine_lines_within_threshold_merge_2(self):
        lines = [1, 47, 50, 98, 100, 450, 500]
        expected = [1, 48, 99, 450, 500]
        new_lines = ps.combine_lines_within_threshold(lines, 6)
        self.assertCountEqual(expected, new_lines)

    def test_combine_lines_within_threshold_merge_3(self):
        lines = [1, 3, 50, 100, 450, 500]
        expected = [2, 50, 100, 450, 500]
        new_lines = ps.combine_lines_within_threshold(lines, 6)
        self.assertCountEqual(expected, new_lines)