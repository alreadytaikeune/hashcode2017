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
    if s_array[i]/float(cols) > 0.54 or s_array[i]/float(cols) < 0.46:
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


def save_solution(best, path=None):
    best = [b for b in best if (b[2], b[3]) != (0, 0)]
    n = len(best)
    if path is None:
        path = "sol_{}.txt".format(path)
    with open(path, "w") as _:
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


def fill_empty_slices(covered_pizza):
    new_slices = []
    for i in range(rows):
        for j in range(cols):
            if covered_pizza[i, j] == 0:
                best_slice = None
                best_slice_score = 0
                for r_length in range(1, max_tot):
                    for w_length in range(1, max_tot/r_length):
                        if i + r_length >= rows:
                            continue
                        if j + w_length >= cols:
                            continue
                        if np.sum(covered_pizza[i:i+r_length, j:j+w_length]) == 0:
                            s_ing = np.sum(pizza[i:i+r_length, j:j+w_length])
                            if s_ing >= min_ing and r_length*w_length - s_ing >= min_ing:
                                print "found potential new slice"
                                # it is a valid slice
                                score = r_length*w_length
                                if score > best_slice_score:
                                    best_slice_score = score
                                    best_slice = (i, j, r_length, w_length)

                if best_slice is not None:
                    new_slices.append(best_slice)
                    x, y, l, w = best_slice
                    covered_pizza[x:x+l,y:y+w] = 1

    print "{} new slices found".format(len(new_slices))
    incr_score = sum([a*b for _, _, a, b in new_slices])
    return new_slices, incr_score


def run_local_optim(final_slices):
    covered_pizza = np.zeros((rows, cols))
    for f in final_slices:
        x, y, l, w = f
        covered_pizza[x:x+l, y:y+w] = 1
    print covered_pizza
    incr_total = 0
    new_slices, incr_total = fill_empty_slices(covered_pizza)
    final_slices.extend(new_slices)
    print "number of slices: {}".format(len(final_slices))
    nb_extended = -1
    while nb_extended != 0:
        refined_sol = []
        nb_extended = 0
        
        for f in final_slices:
            x, y, l, w = f
            if (l+1)*w <= max_tot:
                if x+l < rows:
                    if np.sum(covered_pizza[x+l, y:y+w]) == 0:
                        nb_extended += 1
                        incr_total += w
                        covered_pizza[x+l, y:y+w] = 1
                        l+=1
                    # elif np.sum(covered_pizza[x+l, y:y+w]) < w:
                    #     print "we could have {}".format(np.sum(covered_pizza[x+l, y:y+w]) - w)
            if (w+1)*l <= max_tot:
                if y+w < cols:
                    if np.sum(covered_pizza[x:x+l,y+w]) == 0:
                        nb_extended += 1
                        incr_total += l
                        covered_pizza[x:x+l,y+w] = 1
                        w+=1
                    # elif np.sum(covered_pizza[x:x+l,y+w]) < l:
                    #     print "we could have {}".format(np.sum(covered_pizza[x:x+l,y+w]) - l)
            refined_sol.append((x, y, l, w))
        final_slices = copy.deepcopy(refined_sol)
        print nb_extended
        print incr_total
    print "ameliorated solution by {}".format(incr_total)
    import matplotlib.pyplot as plt
    plt.imshow(covered_pizza, cmap="gray")
    plt.show()
    return final_slices, incr_total


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

# slices = []
# for i in range(1, len(r_crops)):
#     for j in range(1, len(c_crops)):
#         slices.extend(make_slices(r_crops[i]-r_crops[i-1], c_crops[j]-c_crops[j-1], r_crops[i-1], c_crops[j-1]))

slices = make_slices(rows, cols, 0, 0)

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
save_solution(best, "global_big_tmp.txt")
best, incr = run_local_optim(best)
best_score += incr
save_solution(best)
print best_score
e = time.time()
print e-b

