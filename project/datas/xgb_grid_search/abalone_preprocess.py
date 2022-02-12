from copy import copy

if __name__ == '__main__':
    """
    https://archive.ics.uci.edu/ml/datasets/Abalone
    """
    replacement = ""

    with open('abalone.data') as file:
        line = file.readline()
        i = 1
        while line:
            i += 1
            elements = line.split(',')
            elements_copy = copy(elements)
            elements[0] = elements[-1].removesuffix('\n')
            elements[-1] = elements_copy[0].replace('M', '1').replace('F', '0')
            replacement = replacement + ','.join(elements) + "\n"
            line = file.readline()

            if i > 2090:
                break

    with open('sets/abalone_half.csv', 'w') as file:
        file.write(replacement)
