import re
import os
import time
from typing import List
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

from PyPDF2 import PdfFileWriter, PdfFileReader

from pdf2image import convert_from_path
import pytesseract


class Word:
    def __init__(self, word: str, l: int, t: int, r: int, b: int):
        self.word = word
        self.l = l
        self.t = t
        self.r = r
        self.b = b

    def __repr__(self):
        return '{} \t({}, {}, {}, {})'.format(self.word, self.l, self.t, self.r, self.b)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_page_sized_fig(scale=1/25.):
    w, h = 210*scale, 297*scale
    w, h = h, w
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()

    return fig, ax


def plot_words(ax, words):
    for word in words:
        x, y = word.l, word.t
        w, h = word.r - word.l, word.b - word.t
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='k', alpha=1, facecolor='none')
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        ax.annotate(word.word, (cx, cy), color='k', weight='bold',
                    fontsize=5, ha='center', va='center')


def convert_words_to_list(words) -> tuple:
    """ Convert tuple to lists"""
    l = np.array([word.l for word in words])
    r = np.array([word.r for word in words])
    t = np.array([word.t for word in words])
    b = np.array([word.b for word in words])

    return l, t, r, b


def convert_pdf_to_image(pdf: str, dpi=300) -> List[str]:
    pass
    paths = []
    pages = convert_from_path(pdf, dpi=dpi)
    if len(pages) == 1:
        savepath = pdf.replace('.pdf', '.png')
        if not os.path.exists(savepath):
            print("Saving page {} as {}".format(0, savepath))
            pages[0].save(savepath, 'PNG')
        paths.append(savepath)
    else:
        for ii, page in enumerate(pages):
            savepath = pdf.replace('.pdf', '_page%d.png' % ii)
            if not os.path.exists(savepath):
                print("Saving page {} as {}".format(ii, savepath))
                page.save(savepath, 'PNG')
            paths.append(savepath)
    return paths


def get_ocr_output(path: str, lang: str = 'ita') -> List[Word]:
    # Get HOCR output
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print('Starting OCR')
    st = time.time()
    hocr = pytesseract.image_to_pdf_or_hocr(path, lang=lang, extension='hocr')
    print('Finished OCR after %.3f seconds' % (time.time() - st))

    words = []
    root = ET.fromstring(hocr)
    for elem in root.iter():
        if 'class' in elem.attrib:
            cls = elem.attrib['class']
            if cls == 'ocr_line':
                pass
            elif cls == 'ocrx_word':
                coord = elem.attrib['title']
                match = re.search(r'bbox (\d{1,4}) (\d{1,4}) (\d{1,4}) (\d{1,4});', coord)
                if match:
                    l, t, r, b = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
                    words.append(Word(elem.text, l, t, r, b))
    return words


def get_ocr_as_text_output(path: str, lang: str = 'ita', info: bool = False) -> str:
    if info: print('Starting OCR ... ', end='')
    st = time.time()
    string = pytesseract.image_to_string(path, lang=lang, config='--psm 7')
    if info: print('Finished OCR after %.3f seconds' % (time.time() - st))

    return string


def plot_words_and_save(words: List[Word], save_path: str):
    l, t, r, b = convert_words_to_list(words)

    fig, ax = get_page_sized_fig(scale=1/10.)
    # plt.plot([words[0].l, words[0].r], [words[0].t, words[0].b], '-k', lw=2)
    plot_words(ax, words)
    plt.xlim(0, np.max(r) + 5)
    plt.ylim(0, np.max(b) + 5)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    # plt.show()
    plt.savefig(save_path, dpi=300)


def split_pdf(pdf: str):
    inputpdf = PdfFileReader(open(pdf, "rb"))

    for ii in range(inputpdf.numPages):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(ii))
        savepath = pdf.replace('.pdf', '_page%d.pdf' % ii)
        with open(savepath, "wb") as outputStream:
            output.write(outputStream)


def ocr_and_save(file_name: str, suffix: str = ''):
    # words = get_ocr_output(file_name)
    line = get_ocr_as_text_output(file_name)
    load_path = file_name[:-4] + '_{}.txt'.format(suffix)
    with open(load_path, 'r') as f:
        expected = f.read()
    print(line)
    print(expected)


def main():
    data = '/home/edgar/PycharmProjects/OCR/src/test/numbers/'
    pdf = '/home/edgar/OCR/'
    # for file in os.scandir(pdf):
    #     if file.name[-4:] == '.pdf' and 'page' in file.name:
    #         convert_pdf_to_image(file.path)
    # convert_pdf_to_image('/home/edgar/OCR/Scuole_Primarie_1863_page12.pdf')

    import tesserocr
    from tesserocr import PyTessBaseAPI, PSM
    import cv2
    from PIL import Image
    # api = tesserocr.PyTessBaseAPI()

    path = '/home/edgar/OCR/out/Line1_107-169.png'
    img = cv2.imread(path)
    images = ['/home/edgar/OCR/out/Line1_107-169.png', '/home/edgar/OCR/out/Line2_167-229.png']

    with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
        # for img in images:
        api.SetImage(Image.fromarray(img))
        print(api.GetUTF8Text())


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(" --- Ran in {:.3f} seconds --- ".format(time.time() - start_time))
