import os
import math
import statistics
from difflib import SequenceMatcher
import unittest

import src.functions as main


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class TestStringMethods(unittest.TestCase):

    def test_performance(self):
        test_dir = 'ocrtestdatapage9'

        names = []
        ratio = []
        for file in os.scandir(test_dir):
            if file.name[-4:] == '.png':
                data_path = file.path.replace('.png', '_it.txt')
                with open(data_path, 'r') as f:
                    expected = f.read()

                result = main.get_ocr_as_text_output(file.path)
                similarity_ratio = similar(expected, result)
                print("{} -> {}".format(file.name, similarity_ratio))
                names.append(file.name)
                ratio.append(similarity_ratio)

        print("Mean ratio = {}".format(statistics.mean(ratio)))




