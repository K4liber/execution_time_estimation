if __name__ == '__main__':
    """
    https://archive.ics.uci.edu/ml/datasets/Ecoli
    """
    replacement = ""

    with open('ecoli.data') as file:
        line = file.readline()
        i = 1

        while line:
            i += 1
            elements = [el.strip() for el in line.split('  ')][1:]
            elements[-1] = f'"{elements[-1]}"'
            replacement = replacement + ','.join(elements) + "\n"
            line = file.readline()

            if i > 167:
                break

    with open('sets/ecoli_half.csv', 'w') as file:
        file.write(replacement)
