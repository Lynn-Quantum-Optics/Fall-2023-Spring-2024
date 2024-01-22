# file to generalize new_esym_6.py
import numpy as np
from functools import partial

def correlation_classes(d):
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

    return correlation_classes
    
from new_esym_6 import check_orthogonal, check_constraints_and_orthogonality
from oscars_toolbox.trabbit import trabbit
## learning the phases ##
def get_vectors(params):
    '''Assumes each entry is some root of unity. num params per vector is d//2, so total params is (d-1)(d//2)'''

    # extract the dimension
    d = (1+np.sqrt(1+8*len(params)))*0.5
    d = int(d)

    # initialize with the trivial case of no phase
    vectors = [np.ones(d, dtype=complex)]

    # for each remaining vector add the phase
    param_index = 0
    for _ in range(1,d):
        vec_i = [1]
        for j in range(d-1):
            if param_index >= len(params):
                break
            if j > 0 and j%2 == 0:
                # get the phase
                vec_i.append(np.exp(2*np.pi*1j*(params[param_index]+params[eigenval_index])))
                param_index += 1
            else:
                if j==0:
                    eigenval_index = param_index # assign the param that denotes the eigenvalue
                    vec_i.append(np.exp(2*np.pi*1j*params[param_index]))
                    param_index += 1
                else:
                    vec_i.append(np.exp(2*np.pi*1j*params[param_index]))

        vectors.append(np.array(vec_i, dtype=complex))
    
    return vectors

def loss_phase(params, d, guess):
    '''Returns the loss for the phase optimization'''
    if guess:
        # preset some of the params
        guess_params = [(1/2, 0)] # first entry is the eigenvalue
        for i in range(1, d//2):
            guess_params.append((0, i))
        # now have the 0 eigenphase
        for i in range(d//2, (d-2)//2*d//2+1,d//2):
            guess_params.append((0, i))
        # now have the pi eigenphase
        for i in range((d-2)//2*d//2+d//2, (d-2)//2*d//2+d//2+(d-2)//2*d//2, d//2):
            guess_params.append((1/2, i))

        # insert these into params and then generate vectors
        params = list(params)
        for pair in guess_params:
            params.insert(pair[1], pair[0])

        params = np.array(params)

    # get vectors
    vectors = get_vectors(params)

    # get inner products
    inner_prods = 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            inner_prods += np.abs(np.dot(vectors[i], vectors[j].conj().T))

    return inner_prods

def random_gen(num_params):
    '''Generate random parameters'''
    return np.random.uniform(-1, 1, size=(num_params))

def optimize_phase(d, guess = False, tol=1e-10, x0=None):
    '''Minimize the loss function of sum of abs value of inner products to find the optimal phases'''

    # parametrize the funcs
    loss_phase_partial = partial(loss_phase, d=d, guess=guess)
    random_gen_partial = partial(random_gen, num_params=(d-1)*(d//2))


    if x0 is None:
        x_best, loss_best = trabbit(loss_func=loss_phase_partial, random_gen=random_gen_partial, alpha=1, tol=tol, temperature=0.01)
    else:
        x_best, loss_best = trabbit(loss_func=loss_phase_partial, random_gen=random_gen_partial, alpha=1, tol=tol, temperature=0.01, x0_ls=[x0])

    print(f'best loss: {loss_best}')
    print(f'best params: {list(x_best)}')
    return x_best, loss_best

## symbolic ##
import sympy as sp
from new_esym_6 import custom_chop

def get_vectors_symbolic(params):
    '''Same as get_vectors but with sympy'''

    # extract the dimension
    d = (1+np.sqrt(1+8*len(params)))*0.5
    d = int(d)

    # initialize with the trivial case of no phase
    vectors = [sp.Matrix(np.ones(d))]

    # for each remaining vector add the phase
    param_index = 0
    for _ in range(1,d):
        vec_i = [1]
        for j in range(d-1):
            if param_index >= len(params):
                break
            if j > 0 and j%2 == 0:
                # get the phase
                vec_i.append(sp.exp(2*sp.pi*sp.I*(params[param_index]+params[eigenval_index])))
                param_index += 1
            else:
                if j==0:
                    eigenval_index = param_index # assign the param that denotes the eigenvalue
                    vec_i.append(sp.exp(2*sp.pi*sp.I*params[param_index]))
                    param_index += 1
                else:
                    vec_i.append(sp.exp(2*sp.pi*sp.I*params[param_index]))
                    
        vectors.append(sp.Matrix(vec_i))
    
    return vectors

def get_inner_prods(d, numerical_params=None, apply_guess=True, solve=False):
    '''Returns the inner products of the vectors substituting the numerical params if any for even dimension d.

    Params:
        d (int): dimension of system
        numerical_params (list): list of pairs of numerical params to substitute along with the index of the params to substitute
        apply_guess (bool): whether to apply the phase guess or not
        solve (bool): whether to solve for the numerical params or not
    
    '''

    # get symbols
    params = sp.symbols(' '.join([f'p{i}' for i in range((d-1)*(d//2))]))

    # get vectors
    vectors = get_vectors_symbolic(params)
    print('vectors: ', vectors)

    # get inner products
    n = len(vectors)
    results = sp.Matrix.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            results[i, j] = vectors[i].dot(vectors[j].conjugate().T)
            results[j, i] = results[i, j].conjugate()

    # if numerical values are not given, then just print out the expressions
    if numerical_params is None and apply_guess == False:
        for i in range(n):
            print('\Vec{I}'+f'_{i}'+ ' &= \\begin{bmatrix}')
            for j in range(n):
                expr = results[i, j]
                expr = sp.simplify(expr)
                sp.print_latex(expr)
                print('\\\\')
            print('\\end{bmatrix},\\\\')

    else:
        if numerical_params is not None:
            # get list of indices to substitute
            indices = [pair[1] for pair in numerical_params]
            vals = [pair[0] for pair in numerical_params]
            test_params = []
            for i in range(len(params)):
                if i in indices:
                    test_params.append(vals[indices.index(i)])
                else:
                    test_params.append(params[i])

            print('test params: ', test_params)
        elif apply_guess == True:
            # get initial 1, -1, 1, -1, ... vector
            guess_params = [(1/2, 0)] # first entry is the eigenvalue
            for i in range(1, d//2):
                guess_params.append((0, i))
            # now have the 0 eigenphase
            for i in range(d//2, (d-2)//2*d//2+1,d//2):
                print(f'first, {i}')
                guess_params.append((0, i))
            # now have the pi eigenphase
            for i in range((d-2)//2*d//2+d//2, (d-2)//2*d//2+d//2+(d-2)//2*d//2, d//2):
                print(f'second, {i}')
                guess_params.append((1/2, i))

            print('total guess params: ', guess_params)
            # get the complete test_params
            indices = [pair[1] for pair in guess_params]
            vals = [pair[0] for pair in guess_params]
            test_params = []
            for i in range(len(params)):
                if i in indices:
                    test_params.append(vals[indices.index(i)])
                else:
                    test_params.append(params[i])

            print('guess params: ', test_params)

        # apply to results
        results = results.subs({params[i]: test_params[i] for i in range(len(params))})

        for i in range(n):
            print('\Vec{I}'+f'_{i}'+ ' &= \\begin{bmatrix}')
            for j in range(n):
                expr = results[i, j]
                expr = sp.N(expr)
                expr = custom_chop(expr, tol=1e-15)
                expr = sp.simplify(expr)
                sp.print_latex(expr)
                print('\\\\')
            print('\\end{bmatrix},\\\\')

        if solve:
            # solve for values that make sum of abs value of inner products = 0
            total_sum = 0
            for i in range(n):
                for j in range(n):
                    total_sum += sp.Abs(results[i, j])
            total_sum = sp.simplify(total_sum)
            total_sum = [total_sum]
            print(indices)
            # solve for the remaining params
            remaining_params = [params[i] for i in range(len(params)) if i not in indices]

            # initialize the guess
            guess = [0 for _ in range(len(remaining_params))]

            # Solve the equation (if it's solvable)
            solution = sp.nsolve(total_sum,remaining_params, guess)
            print(f'solution: {solution}')
 
def phase_guess(d, neg=False):
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

if __name__ == '__main__':
#    print(sum([
#         -np.cos(np.pi/5) + 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) + 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) + 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) + 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) - 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) - 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) - 1j* np.sin(np.pi/5),
#         -np.cos(np.pi/5) - 1j* np.sin(np.pi/5),
#    ]))

#    print(sum([
#         -np.cos(np.pi/3)+1j * np.sin(np.pi/3),
#         -np.cos(np.pi/3)+1j * np.sin(np.pi/3),
#         -np.cos(np.pi/3)-1j * np.sin(np.pi/3),
#         -np.cos(np.pi/3)-1j * np.sin(np.pi/3),
#    ]))
    # print(get_vectors([0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3]))
    # numerical_params = [(1/3, 1), (-1/3, 2), (1/2, 3), (-1, 4), (0, 5), (2/3, 7), (1/3, 8), (4/3, 10), (-1/3, 11), (2/3, 13), (1/3, 14)]
    # get_inner_prods(10, apply_guess=True)
    optimize_phase(10, tol=1e-10)

