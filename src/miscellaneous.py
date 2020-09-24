import re


def clean_csv(path: str):
    lines = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:

            parts = line.split(',')
            new_part = []

            if parts:
                s = parts[0]
                for x in ['’', '|', '.', ':', '\\', '}', ';', 'î', '‘', ')', '(']:
                    s = s.replace(x, '')
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
        for line in lines:
            f.write('{}\n'.format(','.join(line).replace('\n', '')))
