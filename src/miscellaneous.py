import re
import os
from typing import List

import pdf2image


def clean_csv(path: str):
    lines = []
    with open(path, 'r') as f:
        header = f.readline()
        line = f.readline()
        while line:

            parts = line.split(',')
            new_part = []

            if parts:
                s = parts[0]
                for x in ['’', '|', '.', ':', '\\', '}', ';', 'î', '‘', ')', '(', '!', '[', ']', 'ì']:
                    s = s.replace(x, '')
                s = re.sub(r'\d', '', s)
                new_part.append(s)

                for ii in range(1, len(parts)):
                    s = parts[ii]
                    for x in ['’', '|', '.', ':', '\\', '%', '}', ';']:
                        s = s.replace(x, '')
                    s = s.replace('i', '1').replace('a', '1')

                    match = re.search(r'\d{1,2}?,?\d{1,3}', s)
                    if match:
                        new_part.append(match.group())
                    else:
                        new_part.append(s)
            lines.append(new_part)
            line = f.readline()

    with open(path.replace('.csv', '_cleaned.csv'), 'w') as f:
        f.write(header)
        for line in lines:
            f.write('{}\n'.format(','.join(line).replace('\n', '')))


def convert_pdf_to_image(pdf: str, dpi=300) -> List[str]:
    paths = []
    pages = pdf2image.convert_from_path(pdf, dpi=dpi)
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
