import sys
import numpy as np
import math
import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


path = sys.argv[1]
with open(path, "r") as _:
    rows, cols, min_ing, max_tot = [int(x) for x in _.readline().split(" ")]
    pizza = np.zeros((rows, cols))
    for i, line in enumerate(_.readlines()):
        for j, c in enumerate(line.strip()):
            pizza[i, j] = 1 if c == "M" else 0

print pizza.shape

print float(np.sum(pizza))/(rows*cols)


# plt.plot(range(pizza.shape[0]), np.sum(pizza, axis=0))
# plt.plot(range(pizza.shape[1]), np.sum(pizza, axis=1))
# plt.show()

r_crops = []
s_array = np.sum(pizza, axis=1)
for i in range(pizza.shape[0]):
    if s_array[i]/float(rows) > 0.54 or s_array[i]/float(rows) < 0.46:
        r_crops.append(i)
if r_crops[0] != 0:
    r_crops.insert(0, 0)
r_crops.append(rows)

print len(r_crops)

c_crops = []
s_array = np.sum(pizza, axis=0)
for i in range(pizza.shape[1]):
    if s_array[i]/float(rows) > 0.54 or s_array[i]/float(rows) < 0.46:
        c_crops.append(i)
if c_crops[0] != 0:
    c_crops.insert(0, 0)
c_crops.append(cols)

print len(c_crops)



def list_possible_slices(r_idx, c_idx, mask, pizza):
    out = []
    r_max, c_max = tuple(pizza.shape)
    if mask[r_idx, c_idx] == 1:
        return out
    for i in range(1, min(max_tot, r_max-r_idx)+1):
        for j in range(int(math.ceil(2*min_ing/float(i))), 
            min(int(math.floor(max_tot/float(i))), c_max-c_idx)+1):
            if np.sum(mask[r_idx:r_idx+i, c_idx:c_idx+j]) > 0:
                break
            n_cha = np.sum(pizza[r_idx:r_idx+i,c_idx:c_idx+j])
            n_tom = i*j-n_cha
            if n_tom >= min_ing and n_cha >= min_ing:
                out.append((r_idx, c_idx, i, j))
    return out

def apply_on_mask(s, mask, n):
    i, j, r_length, c_length = s
    mask[i:i+r_length, j:j+c_length] += int(n)

def add_slice(current_slices, new_slice, score, mask):

    current_slices.append(new_slice)
    score += new_slice[2]*new_slice[3]
    apply_on_mask(new_slice, mask, 1)
    # print "add score {}".format(score)
    return score

def remove_slice(current_slices, score, mask):
    last_slice = current_slices.pop()
    score -= last_slice[2]*last_slice[3]
    apply_on_mask(last_slice, mask, -1)
    # print "remove score {}".format(score)
    return score, last_slice


def save_solution(best):
    best = [b for b in best if (b[2], b[3]) != (0, 0)]
    n = len(best)
    with open("sol_{}.txt".format(path), "w") as _:
        _.write(str(n)+"\n")
        for b in best:
            _.write("{} {} {} {}\n".format(b[0], b[1], b[0]+b[2]-1, b[1]+b[3]-1))

# (i, j, r_length, c_length)

def solve_optimal(this_pizza):
    r_max, c_max = tuple(this_pizza.shape)
    current_slices = []
    mask = np.zeros((rows, cols))
    i,j = 0,0
    stack_slices = []
    best_score = 0
    best = []
    score = 0
    if r_max == 0 or c_max == 0:
        return [], score
    while True:
        if i == r_max:
            # print(stack_slices)
            # do something with current_slices
            if score > best_score:
                try:
                    assert(score <= r_max*c_max)
                except AssertionError:
                    import pdb
                    pdb.set_trace()
                # save sol
                best_score = score
                # print "best score {}".format(best_score)
                best = copy.deepcopy(current_slices)
                # save_solution(best)
            # backtrack
            if len(stack_slices) == 0:
                return best, best_score
            while len(stack_slices[-1]) == 0:
                stack_slices.pop()
                if len(stack_slices) == 0:
                    return best, best_score
                score,_ = remove_slice(current_slices, score, mask)

            score, last_slice = remove_slice(current_slices, score, mask)
            i, j = last_slice[0], last_slice[1]
            new_slice = stack_slices[-1].pop()  
            score = add_slice(current_slices, new_slice, score, mask)
            j += max(new_slice[3], 1)
            continue
        if j >= c_max:
            i += 1
            j = 0
            continue
        # list possible slices
        slices = list_possible_slices(i, j, mask, this_pizza)
        if len(slices) == 0:
            j+=1
            continue
        stack_slices.append(slices)
        if not (i*j - np.sum(mask[i, :j])+np.sum(mask[:i, :])) >= r_max*c_max-best_score:
            slices.insert(0, (i, j, 0, 0))
        new_slice = slices.pop()
        score = add_slice(current_slices, new_slice, score, mask)
        j+=max(new_slice[3], 1)


def make_slices(rows, cols, offset_r, offset_c):
    slices = []
    r_step = 6
    c_step = 6
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

slices = []
for i in range(1, len(r_crops)):
    for j in range(1, len(c_crops)):
        slices.extend(make_slices(r_crops[i]-r_crops[i-1], c_crops[j]-c_crops[j-1], r_crops[i-1], c_crops[j-1]))


tmp = np.zeros((rows, cols))
for s in slices:
    i, j, r_length, c_length = s
    assert(np.sum(tmp[i:i+r_length, j:j+c_length]) == 0)
    tmp[i:i+r_length, j:j+c_length] = 1
assert(np.sum(tmp) == rows*cols)


parallel = True
import time
b = time.time()
if parallel:
    res = Parallel(n_jobs=4)(delayed(solve_optimal)(pizza[i:i+r_length, j:j+c_length]) for i, j, r_length, c_length in slices)
    best = []
    best_score = 0
    for i, (bs, s) in enumerate(res):
        for bb in bs:
            if bb[2] != 0:
                best.append((bb[0]+slices[i][0], bb[1]+slices[i][1], bb[2], bb[3]))
        best_score += s
else:
    best,best_score = solve_optimal(pizza)
print best_score
save_solution(best)
print best_score
e = time.time()
print e-b

