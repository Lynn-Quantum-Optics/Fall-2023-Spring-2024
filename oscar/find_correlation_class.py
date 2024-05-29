''' 
file to generate finding correlation classes using recursive backtracking for the paper Scholin, O. and Lynn, T.W. 2024. "Maximal Limits on Distinguishing Bell States with $d$-dimensional Single Particles by Linear Evolution and Local Projective Measurement". Submitted to Physical Review A. 

Author: Oscar Scholin
'''

def get_correlation_classes(d, print_result=False):
    '''Returns the correlation classes for a given d. Assumes d is even.'''

    assert d % 2 == 0, f'd must be even. Your d is {d}'

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

    if print_result:
        for i in range(d):
            print(f'correlation class {i}: {correlation_classes[i]}')

    return correlation_classes

if __name__ == '__main__':
    d = 4
    get_correlation_classes(4, print_result=True)