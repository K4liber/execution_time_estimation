from copy import copy

if __name__ == '__main__':
    """
    https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
    """
    replacement = ""

    with open('sets/segmentation.data') as file:
        line = file.readline()
        i = 1

        while line:
            i += 1
            elements = [el.strip() for el in line.split(',')]
            elements_copy = copy(elements)
            elements[0] = elements[-1]
            elements[-1] = elements_copy[0]
            elements[-1] = f'"{elements[-1]}"'
            replacement = replacement + ','.join(elements) + "\n"
            line = file.readline()

            if i > 104:
                break

    with open('sets/segmentation_half.csv', 'w') as file:
        file.write(replacement)
