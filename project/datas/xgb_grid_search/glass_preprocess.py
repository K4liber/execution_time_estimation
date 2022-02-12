if __name__ == '__main__':
    """
    https://archive.ics.uci.edu/ml/datasets/Glass+Identification
    """
    replacement = ""

    with open('glass.data') as file:
        line = file.readline()
        i = 1

        while line:
            i += 1
            elements = list(line.split(','))[1:]
            elements[-1] = elements[-1].removesuffix('\n')
            replacement = replacement + ','.join(elements) + "\n"
            line = file.readline()

            if i > 106:
                break

    with open('sets/glass_half.csv', 'w') as file:
        file.write(replacement)
