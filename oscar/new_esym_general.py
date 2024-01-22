# file to generalize new_esym_6.py
import numpy as np
from copy import deepcopy

def OLD_correlation_classes(d):
    '''Returns correlation classes for a given even d'''

    assert d % 2 == 0, f'd must be even. Your d is {d}'

    def valid_cc(correlation_i):
        '''Returns whether a correlation class is valid or not.'''
        # get all pairs
        pairs = []
        for pair in correlation_i:
            pairs.append(pair)
            pairs.append((pair[1], pair[0]))
        # check that all pairs are unique
        if len(pairs) != len(set(pairs)):
            return False
        # check that all pairs are of length d
        if len(pairs) != d:
            return False
        # check that there are no repeated elements
        for pair in pairs:
            if pair[0] == pair[1]:
                return False
        return True

    # get all correlation classes
    correlation_classes = [[] for _ in range(d)]
    for i in range(d):
        correlation_classes[0].append((i, i)) # add trivial correlation class

    # get all correlation classes
    used_pairs = []
    for i in range(1,d):
        correlation_i = []
        exit_outer_loop = False  # flag for whether to exit the outer loop
        for j in range(d):
            if exit_outer_loop:
                break
            for k in range(d):
                if len(correlation_i) == d:
                    # append to correlation classes
                    correlation_classes[i] = correlation_i
                    exit_outer_loop = True  # exit to i loop
                    break
                if j != k:
                    # get pair and reverse
                    pair = (j, k)
                    rev_pair = (k, j)
                    # see if pair is already used
                    if (pair not in used_pairs and rev_pair not in used_pairs) and len(correlation_i) < d:
                        # print(f'pair = {pair}')
                        # to ensure entanglement, we need to make sure that the neither j nor k are in the correlation_i
                        if len(correlation_i) == 0:
                            print(f'len 0, adding to correlation_{i}: {pair}')
                            # add pair and pair reversed
                            correlation_i.append(pair)
                            correlation_i.append(rev_pair)
                            # add to used pairs
                            used_pairs.append(pair)
                            used_pairs.append(rev_pair)
                        else:
                            # make sure that neither j nor k are in correlation_i for entanglement
                            for p in correlation_i:
                                print(f'pair inside = {p}, {j}, {k}')
                                no_element_in_p = True
                                if j in p or k in p:
                                    no_element_in_p = False
                                    break
                            # now confirm that if we add this pair, there is a valid solution to the end
                            test_correlation_i = correlation_i + [pair, rev_pair]
                            # from wherever j, k are continue iterating
                            for l in range(j, d):
                                for m in range(d):
                                    if l != m:
                                        # get pair and reverse
                                        test_pair = (l, m)
                                        test_rev_pair = (m, l)
                                        # see if pair is already used
                                        if (test_pair not in used_pairs and test_rev_pair not in used_pairs) and len(test_correlation_i) < d:
                                            pass
                                          














                            if no_element_in_p:
                                print(f'adding to correlation_{i}: {pair}')
                                # add pair and pair reversed
                                correlation_i.append(pair)
                                correlation_i.append(rev_pair)
                                # add to used pairs
                                used_pairs.append(pair)
                                used_pairs.append(rev_pair)
                                print(f'len correlation_{i} = {len(correlation_i)}, {j}, {k}')
                                if len(correlation_i) == d:
                                    # append to correlation classes
                                    correlation_classes[i] = correlation_i
                                    exit_outer_loop = True

    # check that all correlation classes are of length d
    for i in range(d):
        print(f'correlation class {i}: {correlation_classes[i]}')
        assert len(correlation_classes[i]) == d, f'Correlation classes are not of length d. Your correlation class {i} is {correlation_classes[i]}'

def correlation_classes(d):
     # get all correlation classes
    correlation_classes = [[] for _ in range(d)]
    for i in range(d):
        correlation_classes[0].append((i, i)) # add trivial correlation class

    # seed correlation classes
    used_pairs = set([])
    for i in range(1, d):
        correlation_classes[i].append((0, i))
        correlation_classes[i].append((i, 0))
        used_pairs.add((0, i))
        used_pairs.add((i, 0))

    def is_valid_cc(correlation):
        '''Checks whether for this correlation class, everything is entangled'''
        # check total length is d
        if len(correlation) != d:
            return False
        
        # get all pairs
        L_ls = []
        R_ls = []
        for pair in correlation:
            L_ls.append(pair[0])
            R_ls.append(pair[1])
            # confirm that the reversed version is also in the correlation
            assert (pair[1], pair[0]) in correlation, f'Your correlation class {correlation} is not valid. The reverse of pair {pair} is not in the correlation class.'

        # check that all pairs are unique
        if len(L_ls) != len(set(L_ls)) or len(R_ls) != len(set(R_ls)):
            return False
        
        return True
        
    def fill_in_remaining(correlation):
        '''Is it possible to fill in the remaining pairs?'''

        # iterate from k, l to d
        for m in range(d):
            for n in range(d):
                if m != n:
                    pair = (m, n)
                    rev_pair = (n, m)

                    # check if the pair or its reverse is not used
                    if pair not in used_pairs and rev_pair not in used_pairs:
                        # Ensure neither m nor n are in the correlation
                        no_element_in_correlation = True
                        for p in correlation:
                            if m in p or n in p:
                                no_element_in_correlation = False
                                break
                        
                        if no_element_in_correlation:
                            # Add pair to correlation and used pairs
                            correlation.append(pair)
                            correlation.append(rev_pair)
                            used_pairs.add(pair)
                            used_pairs.add(rev_pair)

                            # Recursively try to fill in remaining
                            if fill_in_remaining(correlation) or is_valid_cc(correlation):
                                return True
                        
                            # Backtrack
                            correlation.pop()
                            correlation.pop()
                            used_pairs.remove(pair)
                            used_pairs.remove(rev_pair)


        # If all pairs are exhausted
        return False

    # find the solutions
    for i in range(1, d):
        correlation_i = correlation_classes[i]
        # fill in remaining
        fill_in_remaining(correlation_i)

    # check that all correlation classes are of length d
    for i in range(d):
        # print(f'correlation class {i}: {correlation_classes[i]}')
        assert len(correlation_classes[i]) == d, f'Correlation classes are not of length d. Your correlation class {i} is {correlation_classes[i]}'

    return correlation_classes
    




def phase(d, neg=False):
    # # get n
    # n = d - 2
    # m = n-1

    # # get first half above
    # exp_ls = []
    # for i in range(n//2):
    #     if i == 0:
    #         exp_ls.append(np.pi/m)
    #     else:
    #         # get last entry
    #         exp_ls.append(np.pi/m + exp_ls[-1])
    # # add pi - pi/n to last entry
    # for i in range(n//2):
    #     if i == 0:
    #         exp_ls.append(np.pi - np.pi/m + exp_ls[-1])
    #     else:
    #         exp_ls.append(np.pi/m + exp_ls[-1])

    # # get exponents
    # return sum(np.exp(1j * np.array(exp_ls)))

    m = d//2

    if neg == False:
        phase_ls = [1, 1] + [np.exp(np.pi*(1-1/m)*1j)]*((d-2)//2) + [np.exp(np.pi*(1+1/m)*1j)]*((d-2)//2)

    else:
        # print(np.array([np.exp(np.pi*1j*i/m) for i in range(1, m)] + [np.exp(np.pi*1j*(m+i)/m) for i in range(1, m)]))
        phase_ls = [1, -1] + list(np.array([np.exp(np.pi*1j*(j%2)) for j in range(2, d)])*np.array([np.exp(np.pi*1j/m)]*((d-2)//2) + [np.exp(np.pi*1j*(m+m-1)/m)]*((d-2)//2)))

    return np.array(phase_ls)

d = 18
# print(phase(d, neg=False)@np.ones(d).T)
correlation_classes(d)

