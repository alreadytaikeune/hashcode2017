import pulp
import sys
import numpy as np
import math
from joblib import Parallel, delayed

def is_slice_valid(p_slice):
    c, r = tuple(p_slice.shape)
    s = np.sum(p_slice)
    return s>=min_ing and s <= c*r-min_ing
    
path = sys.argv[1]

with open(path, "r") as _:
    rows, cols, min_ing, max_tot = [int(x) for x in _.readline().split(" ")]
    pizza = np.zeros((rows, cols))
    for i, line in enumerate(_.readlines()):
        for j, c in enumerate(line.strip()):
            pizza[i, j] = 1 if c == "M" else 0

print pizza.shape

valid_slices_sizes = []
for i in range(1, max_tot+1):
    j_min = int(math.ceil(min_ing*2/float(i))) # inclusive
    j_max = int(max_tot/float(i)) + 1 # exclusive
    for j in range(j_min, j_max):
        valid_slices_sizes.append((i, j))
        if j != i:
            valid_slices_sizes.append((j, i))
# print valid_slices_sizes

def solve_optimal(pizza):
    rows, cols = tuple(pizza.shape)
    possible_slices = []
    weights = []
    in_values = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            for shape in valid_slices_sizes:
                r_length, c_length = shape
                if i + r_length <= rows and j + c_length <= cols:
                    if is_slice_valid(pizza[i:i+r_length, j:j+c_length]):
                        possible_slices.append((i, j, r_length, c_length))
                        in_values.append(count)
                        count += 1
                        weights.append(r_length*c_length)
    # possible_slices = [(0, 0, 3, 2), (0,1,3,1), (0, 2, 3, 2)]
    print "there are {} possible slices".format(len(possible_slices))


    x = pulp.LpVariable.dicts('slices', in_values, 
                                lowBound = 0,
                                upBound = 1,
                                cat = pulp.LpInteger)

    slice_model = pulp.LpProblem("Slice cutting model", pulp.LpMaximize)

    slice_model += sum([weights[s]*x[s] for s in in_values])

    for r in range(rows):
        for c in range(cols):
            slice_model += sum([x[s] for s in in_values \
                if r >= possible_slices[s][0] and r < possible_slices[s][0]+possible_slices[s][2] \
                and c >= possible_slices[s][1] and c < possible_slices[s][1]+possible_slices[s][3]]) <= 1, "must be included once {} {}".format(r, c)

    slice_model.solve()
    print "Status:", pulp.LpStatus[slice_model.status]
    sol = []
    score = 0
    nb = 0
    for s in in_values:
        # print x[s].value()
        if x[s].value() == 1.0:
            sol.append(possible_slices[s])
            nb += 1
            score += possible_slices[s][2]*possible_slices[s][3]
    print nb
    return sol, score

def save_solution(best, path=None):
    best = [b for b in best if (b[2], b[3]) != (0, 0)]
    n = len(best)
    if path is None:
        path = "sol_{}.txt".format(path)
    with open(path, "w") as _:
        _.write(str(n)+"\n")
        for b in best:
            _.write("{} {} {} {}\n".format(b[0], b[1], b[0]+b[2]-1, b[1]+b[3]-1))


def make_slices(rows, cols, offset_r, offset_c):
    slices = []
    r_step = 10
    c_step = 10
    for i in range(0, rows, r_step):
        r_length = r_step
        if i + r_length >= rows:
            r_length = rows-i
        for j in range(0, cols, c_step):
            c_length = c_step
            if j + c_length >= cols:
                c_length = cols-j
            slices.append((i+offset_r, j+offset_c, r_length, c_length))
    return slices

slices = make_slices(rows, cols, 0, 0)

# res = [solve_optimal(pizza[i:i+r_length, j:j+c_length]) for i, j, r_length, c_length in slices]
res = Parallel(n_jobs=4)(delayed(solve_optimal)(pizza[i:i+r_length, j:j+c_length]) for i, j, r_length, c_length in slices)
best = []
best_score = 0
for i, (bs, s) in enumerate(res):
    for bb in bs:
        if bb[2] != 0:
            best.append((bb[0]+slices[i][0], bb[1]+slices[i][1], bb[2], bb[3]))
    best_score += s

# best, best_score = solve_optimal(pizza)
print best_score
save_solution(best, "mip_{}.txt".format(path))