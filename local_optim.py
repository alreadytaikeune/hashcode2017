import numpy as np
import sys
import copy

path = sys.argv[1]
is_google = (len(sys.argv) == 6)
if is_google:
    rows, cols, min_ing, max_tot = [int(x) for x in sys.argv[2:6]]
else:
    cols, rows, min_ing, max_tot = [int(x) for x in sys.argv[2:6]]
final_slices = []
with open(path, "r") as _:
    pizza = np.zeros((rows, cols))
    print "{} slices".format(_.readline())
    for i, line in enumerate(_.readlines()):
        if is_google:
            x1, y1, x2, y2 = [int(x) for x in line.split(" ")]
        else:
            x1, y1, x2, y2 = [int(x) for x in line.split(",")]
        pizza[x1:x2+1, y1:y2+1] = 1
        if is_google:
            final_slices.append((x1, y1, x2-x1+1, y2-y1+1))
        else:
            final_slices.append((x1, y1, x2, y2))


def save_solution(best, path=None):
    best = [b for b in best if (b[2], b[3]) != (0, 0)]
    n = len(best)
    if path is None:
        path = "sol_{}.txt".format(path)
    with open(path, "w") as _:
        _.write(str(n)+"\n")
        for b in best:
            _.write("{} {} {} {}\n".format(b[0], b[1], b[2], b[3]))

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

best, incr_score = run_local_optim(final_slices)
print "incremented by {}".format(incr_score)
save_solution(best, "local_optim_{}.txt".format(path))