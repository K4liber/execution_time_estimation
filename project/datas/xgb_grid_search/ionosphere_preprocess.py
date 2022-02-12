if __name__ == '__main__':
    """
    https://archive.ics.uci.edu/ml/datasets/Ionosphere
    """
    replacement = ""

    with open('ionosphere.data') as file:
        line = file.readline()
        i = 1

        while line:
            i += 1
            elements = [el.strip() for el in line.split(',')]
            elements[-1] = f'"{elements[-1]}"'
            replacement = replacement + ','.join(elements) + "\n"
            line = file.readline()

            if i > 175:
                break

    with open('sets/ionosphere_half.csv', 'w') as file:
        file.write(replacement)
