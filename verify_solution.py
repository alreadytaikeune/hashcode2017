import sys
import numpy as np 
path = sys.argv[1]
rows, cols, min_ing, max_tot = [int(x) for x in sys.argv[2:]]
arr = np.zeros((rows, cols))
with open(path, "r") as _:
    _.readline()
    for line in _.readlines():
        x1, y1, x2, y2 = [int(x) for x in line.split(" ")]

        if (x2-x1+1)*(y2-y1+1) > max_tot:
            raise ValueError()
        if np.sum((arr[x1:x2+1, y1:y2+1])) > 0:
            raise ValueError("clash")
        arr[x1:x2+1, y1:y2+1] = 1

