import csv
import random
import time
import numpy as np


def random_vector(dim):
    return [np.float32(random.random()) for _ in range(dim)]


def main():
    out = list()

    with open("/home/yhz/Downloads/ml-latest-small/movies.csv", 'r') as f:
        cr = csv.reader(f)
        offset = 0
        for i, l in enumerate(cr):
            print(i)
            if i == 0:
                continue

            names = l[1]
            if names.count("(") != 1:
                continue

            names = names.strip('"')
            name_list = names.split("(")
            name = name_list[0].strip()
            year = int(name_list[1][: 4])
            ol = [offset, name, year, random_vector(8)]
            print(ol)
            out.append(ol)
            offset += 1

    with open("films.csv", "w") as f:
        sw = csv.writer(f)
        for o in out:
            sw.writerow(o)


if __name__ == '__main__':
    main()
