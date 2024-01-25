# file to generalize new_esym_6.py
import numpy as np
from functools import partial

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
    
from new_esym_6 import check_orthogonal, check_constraints_and_orthogonality
from oscars_toolbox.trabbit import trabbit
## learning the phases ##
def get_vectors(params):
    '''Assumes each entry is some root of unity. num params per vector is d//2, so total params is (d-1)(d//2)'''

    # extract the dimension
    d = (1+np.sqrt(1+8*len(params)))*0.5
    d = int(d)

    print(f'params: {params}')

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

def loss_phase(params, guess=False, replacement_params=None, print_out=False):
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

    if replacement_params is not None:
        new_numerical_params = []
        for i in range(len(params)):
            elem = params[i]
            if type(elem)==float: 
                new_numerical_params.append(elem)
            else:
                new_numerical_params.append(replacement_params[i])
        params = new_numerical_params

    # get vectors
    vectors = get_vectors(params)

    # get inner products
    inner_prods = 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            abs_ij = np.abs(np.dot(vectors[i], vectors[j].conj().T))
            if print_out:
                print(f'inner product {i}, {j}: {abs_ij}')
            inner_prods += abs_ij

    return inner_prods

def sum_abs_inner_prods(params, print_out=False):
    '''Assumes list of tuples of (phase, index)'''
    # get ordered params to sent to loss_phase
    ordered_params = [0 for _ in range((d-1)*(d//2))]
    for pair in params:
        ordered_params[pair[1]] = pair[0]

    # get vectors
    return loss_phase(ordered_params, print_out=print_out)



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

def get_det(params):
    # get vectors
    vectors = get_vectors(params)
    # create a matrix where the column vectors are the vectors
    result_matrix = np.zeros((vectors[0].shape[0], 0), dtype=complex)

    # Concatenate each matrix in the list as a new column
    for mat in vectors:
        result_matrix = np.hstack((result_matrix, mat.reshape(-1, 1)))

    # print each row separately
    for row in result_matrix:
        # add & signs between each entry
        for i in range(len(row)):
            if i == len(row)-1:
                print(np.round(row[i],3), end='\\\\')
            else:
                print(np.round(row[i],3), end=' & ')
    print('----')

    # take the inner product of all the columns
    for i in range(result_matrix.shape[1]):
        for j in range(i+1, result_matrix.shape[1]):
            print(np.round(np.abs(np.dot(result_matrix[:, i], result_matrix[:, j].conj())),3), end=' & ')
        print('\\\\')

    # get determinant
    det_result  = np.linalg.det(result_matrix)
    print('determinant: ', det_result)

    # take V^dagger V
    VdaggerV = result_matrix.conj().T @ result_matrix
    print('----')
    # print each row separately
    for row in VdaggerV:
        # add & signs between each entry
        for i in range(len(row)):
            if i == len(row)-1:
                print(np.round(row[i],3), end='\\\\')
            else:
                print(np.round(row[i],3), end=' & ')
    # get determinant
    det_result  = np.linalg.det(VdaggerV)
    print(f'determinant VdaggerV:  {det_result}')

    
    # return det_result

def get_inner_prod_sub(d, params, num_vecs = None):
    '''Computes inner product of subset of total vectors for a given set of params'''

    if num_vecs is None:
        num_vecs = d

    # get vectors
    vectors = get_vectors(params)

    # get inner products
    inner_prods = 0
    for i in range(num_vecs):
        for j in range(i+1, num_vecs):
            inner_prods += np.abs(np.dot(vectors[i], vectors[j].conj().T))

    print(f'inner prods for {num_vecs} vectors: {inner_prods}')
    return inner_prods

## symbolic ##
import sympy as sp
from new_esym_6 import custom_chop

def get_vectors_symbolic(params):
    '''Same as get_vectors but with sympy'''

    # extract the dimension
    try:
        d = (1+np.sqrt(1+8*len(params)))*0.5
        d = int(d)
    except TypeError:
        vectors = [sp.Matrix(np.ones(2))]
        vec_i = [1, sp.exp(2*sp.pi*sp.I*params)]
        vectors.append(sp.Matrix(vec_i))
        return vectors
    

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

    print('vectors: ', vectors)
    
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
    params = sp.symbols(' '.join([f'p{i}' for i in range((d-1)*(d//2))]), real=True)

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
                # print([i+1+j*d//2 for i in range(d//2)])
                print('\\\\')
            print('\\end{bmatrix},\\\\')

    else:
        if numerical_params is not None:
            # only keep the elements with a pair in numerical_params
            new_numerical_params = []
            for i in range(len(numerical_params)):
                elem = numerical_params[i]
                if type(elem)!=float: 
                    new_numerical_params.append(elem)

            numerical_params = new_numerical_params
            
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
        results_old = results
        results = results.subs({params[i]: test_params[i] for i in range(len(params))})

        for i in range(n):
            print('\Vec{I}'+f'_{i}'+ ' &= \\begin{bmatrix}')
            for j in range(n):
                expr = results[i, j]
                expr = sp.N(expr)
                expr = custom_chop(expr, tol=1e-15)
                expr = sp.simplify(expr)
                sp.print_latex(expr)
                if expr != 0:
                    print('expr old')
                    print(results_old[i, j])
                    # print([i+k*d//2 for k in range(d-2)])
                    # print([j+k*d//2 for k in range(d-2)])
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

def simplify_system(d, numerical_params):
    '''Takes in a list of params, creates vectors, and returns the system of equations given those values'''

    # get symbols
    params = sp.symbols(' '.join([f'p{i}' for i in range((d-1)*(d//2))]), real=True)

    # get vectors
    vectors = get_vectors_symbolic(params)

    # only keep the elements with a pair in numerical_params
    new_numerical_params = []
    for i in range(len(numerical_params)):
        elem = numerical_params[i]
        if type(elem)!=float: 
            new_numerical_params.append(elem)

    numerical_params = new_numerical_params
    
    # get list of indices to substitute
    indices = [pair[1] for pair in numerical_params]
    vals = [pair[0] for pair in numerical_params]
    test_params = []
    for i in range(len(params)):
        if i in indices:
            test_params.append(vals[indices.index(i)])
        else:
            test_params.append(params[i])

    # substitute into vectors
    vectors = [vec.subs({params[i]: test_params[i] for i in range(len(params))}) for vec in vectors]

    # now list out all unique inner products
    n = len(vectors)
    for i in range(n):
        for j in range(i+1, n):
            expr = vectors[i].dot(vectors[j].conjugate().T)
            expr = sp.N(expr)
            expr = custom_chop(expr, tol=1e-15)
            expr = sp.simplify(expr)
            if expr != 0:
                # print(f'inner product {i}, {j}: ')
                sp.print_latex(expr)
                print('\\\\')  

def row_reduce(d):
    '''Computes the row reduced form of the matrix of vectors to test linear independence'''
    # get the params
    params = sp.symbols(' '.join([f'p{i}' for i in range((d-1)*(d//2))]), real=True)

    # get vectors
    vectors = get_vectors_symbolic(params)

    # create a matrix where the column vectors are the vectors
    # Initialize an empty matrix with the same number of rows as the matrices in your list
    result_matrix = sp.Matrix(vectors[0].shape[0], 0, [])

    # Concatenate each matrix in the list as a new column
    for mat in vectors:
        result_matrix = result_matrix.row_join(mat)

    sp.print_latex(result_matrix)

    # get determinant
    det_result  = result_matrix.det()
    print('determinant: ', sp.print_latex(det_result))
    # get random params
    random_params = np.random.uniform(-1, 1, size=(d-1)*(d//2))
    det_result = det_result.subs({params[i]: random_params[i] for i in range(len(params))})
    # evaluate
    det_result = sp.N(det_result)
    print('determinant numerical: ', sp.print_latex(det_result))
    # print('determinant simplified: ', sp.print_latex(sp.simplify(det_result)))
    
def compute_vandermond_det(d, params):
    '''Computes determinant of wronskian of p_vec'''
    # get all, p_j, p_{j+1}, p_{j}+p_{j+2}, p_{j+1}+p_{j+3}, ... p_{j}+p_{j+d/2}
    alpha = params[0]
    vander_param = [alpha]
    param_index = 1
    for _ in range(1,d):
        for j in range(d-1):
            if param_index >= len(params):
                break
            if j > 0 and j%2 == 0:
                # get the phase
                vander_param.append(alpha + params[param_index]+params[eigenval_index])
                param_index += 1
            else:
                if j==0:
                    eigenval_index = param_index # assign the param that denotes the eigenvalue
                    vander_param.append(alpha + params[param_index])
                    param_index += 1
                else:
                    vander_param.append(alpha + params[param_index])
                    
    # compute product of c_j - c_i fpr i < j
    print(params)
    prod = 1
    for i in range(len(vander_param)):
        for j in range(i):
            prod *= (vander_param[j] - vander_param[i])

    print('vandermonde determinant: ', prod)
    return prod
            
def show_orthogonality(d):
    '''Compute V^T V'''
    # get the params
    params = sp.symbols(' '.join([f'p{i}' for i in range((d-1)*(d//2))]), real=True)

    # get vectors
    vectors = get_vectors_symbolic(params)

    # create a matrix where the column vectors are the vectors
    # Initialize an empty matrix with the same number of rows as the matrices in your list
    result_matrix = sp.Matrix(vectors[0].shape[0], 0, [])

    # Concatenate each matrix in the list as a new column
    for mat in vectors:
        result_matrix = result_matrix.row_join(mat)

    # take V^dagger V
    result_matrix = result_matrix.H* result_matrix
    sp.print_latex(result_matrix)

# def create_orthogonal_vectors(d):
#     # Number of columns = (d - 2) / 2
#     num_cols = (d - 2) // 2

#     # Number of rows = (d - 2) / 2
#     num_rows = (d - 2) // 2

#     # Create a modified DFT matrix
#     n = np.arange(1, d//2+1)
#     k = np.arange(1, num_rows+1).reshape((num_rows, 1))
#     omega =  k * n[:num_cols] / d

#     return omega
    
def create_orthogonal_vectors(d):
    # Create a modified DFT matrix
    n = np.arange(d//2)
    k = n.reshape((d//2, 1))
    omega = k * n / (d//2)

    # remove first row and column
    omega = omega[1:, 1:]
    return omega

def create_tuples_from_vectors(vectors, d):
    print(vectors)
    # Reshape into matrix M
    M = vectors.reshape(((d - 2) // 2, (d - 2) // 2))

    # Create a list of tuples
    tuples = []
    # Increment value for indices
    increment = d // 2

    for i in range(M.shape[0]):
        # Handle the first element of each row
        tuples.append((M[i][0], i + 1))
        tuples.append((M[i][0], i + 1 + 3 * increment))
        # Special case for the middle index
        tuples.append((0, i + 1 + increment))
        # next pair
        tuples.append((M[i][1], i + 1 + 2 * increment))
        tuples.append((M[i][1], i + 1 + 4 * increment))

        if M.shape[1] > 2:
            # Handle subsequent elements of each row
            offset = 0
            for j in range(2,M.shape[1]):
                if j > 2 and j % 2 == 0:
                    offset += 2
                int_j = j + 2 + offset
                index1 = i + 1 + increment + int_j * increment
                
                index2 = index1 + 2*increment
                # check if index2 is out of bounds
                if index2 > d**2//2 - d//2:
                    index2 -=  increment

                tuples.append((M[i][j], index1))
                tuples.append((M[i][j], index2))

    return tuples

if __name__ == '__main__':
    # get_correlation_classes(6, print_result=True)

    # create param
    d = 14
   
    # append create tuples from vectors
    vectors = create_orthogonal_vectors(d)
    tuples = create_tuples_from_vectors(vectors, d)

    # append to numerical params
    numerical_params = tuples

    # numerical_params =[]
    # first add the eigenphases
    numerical_params+=[(1/2, d//2), (1/2, 0), (0, 3*d//2)]
    numerical_params+=[(1/2, 2*d//2), (0,4*d//2)] # first in main
    numerical_params+=[(1/2, (d**2//2 - 3*d//2)), (0, (d**2//2 - d))] # last in main

    current_5 = 5
    current_0 = 7
    for i in range(5, d//2+1):
        print(current_5*(d//2))
        numerical_params.append((1/2, current_5*(d//2)))
        numerical_params.append((0, current_0*(d//2)))
        current_5 += 1
        current_0 += 1



       

    
   

    # print(numerical_params)

    # sort the numerical params
    # numerical_params = sorted(numerical_params, key=lambda x: x[1])


    # correct_12 = [(1/2, 6), (1/2, 12), (0, 24), (1/2, 30), (0, 42), (1/2, 36), (0, 48), (1/2, 54), (0, 60), (1/2, 0), (0, 18)]

    # # sort
    # correct_12 = sorted(correct_12, key=lambda x: x[1])

    # # compare
    # print(numerical_params)
    # print(correct_12)

    # print(sum_abs_inner_prods(numerical_params))
        

        

    # numerical_params.append((0, d))
    # numerical_params.append((1/2, 24))
    # if d == 12:
    #     numerical_params.append((1/2,36))
    #     # numerical_params.append((0,48))
    #     # numerical_params.append((1/2,24))
    #     # numerical_params.append((1/2,30))
    #     numerical_params.append((0,60))
    # if d == 10:
    #     numerical_params.append((1/2, 40))
    #     numerical_params.append((0, 25))

    ## for d = 6
    # for i in range(d-1):
    #     if i == d-2:
    #         numerical_params.append((1/2, i*d//2))
    #     elif i %2 == 0:
    #         numerical_params.append((0, i*d//2))
    #     elif i %2 != 0:
    #         numerical_params.append((1/2, i*d//2))
    #     else:
    #         numerical_params.append((1/2, i*d//2))
 

    

    # compute inner products
    get_inner_prods(d, numerical_params=numerical_params)
    # sum([np.exp(np.pi)])
    
    # print(numerical_params)
    # print(sum_abs_inner_prods(numerical_params, print_out=True))

    # print(sum([np.exp(2*np.pi*1j/3), 1, np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3), np.exp(4*np.pi*1j/3)]))
        
    # params = [(0.16666666666666666, 1), (0.16666666666666666, 19), (0, 7), (0.3333333333333333, 13), (0.3333333333333333, 25), (0.5, 31), (0.5, 43), (0.6666666666666666, 37), (0.6666666666666666, 49), (0.8333333333333334, 43), (0.8333333333333334, 55), (0.3333333333333333, 2), (0.3333333333333333, 20), (0, 8), (0.6666666666666666, 14), (0.6666666666666666, 26), (1.0, 32), (1.0, 44), (1.3333333333333333, 38), (1.3333333333333333, 50), (1.6666666666666667, 44), (1.6666666666666667, 56), (0.5, 3), (0.5, 21), (0, 9), (1.0, 15), (1.0, 27), (1.5, 33), (1.5, 45), (2.0, 39), (2.0, 51), (2.5, 45), (2.5, 57), (0.6666666666666666, 4), (0.6666666666666666, 22), (0, 10), (1.3333333333333333, 16), (1.3333333333333333, 28), (2.0, 34), (2.0, 46), (2.6666666666666665, 40), (2.6666666666666665, 52), (3.3333333333333335, 46), (3.3333333333333335, 58), (0.8333333333333334, 5), (0.8333333333333334, 23), (0, 11), (1.6666666666666667, 17), (1.6666666666666667, 29), (2.5, 35), (2.5, 47), (3.3333333333333335, 41), (3.3333333333333335, 53), (4.166666666666667, 47), (4.166666666666667, 59), (0, 0), (0.5, 6), (0, 12), (0.5, 18), (0, 24), (0.5, 30), (0, 36), (0.5, 42), (0, 48), (0.5, 54), (0.5, 60), (0, 12), (0.5, 36), (0, 48), (0.5, 24), (0.5, 30)]


    # p36 = 0.5
    # p41 = 0.5
    # p48 = 0.5
    # p53 = 0.5
    # p40 = 0.5
    # p52 = 0.5
    # p39 = 0.5
    # p51 = 0.5
    # p38 = 0.5
    # p50 = 0.5
    # p37 = 0.5


        

    # val = 1 + np.exp(-2*1j*np.pi*(p36 + p41))*np.exp(2*1j*np.pi*(p48 + p53)) + np.exp(-2*1j*np.pi*(p36 + p40))*np.exp(2*1j*np.pi*(p48 + p52)) + np.exp(-2*1j*np.pi*(p36 + p39))*np.exp(2*1j*np.pi*(p48 + p51)) + np.exp(-2*1j*np.pi*(p36 + p38))*np.exp(2*1j*np.pi*(p48 + p50)) + np.exp(-2*1j*np.pi*(p36 + p37))*np.exp(2*1j*np.pi*(p48 + p49)) + np.exp(-2*1j*np.pi*p41)*np.exp(2*1j*np.pi*p53) + np.exp(-2*1j*np.pi*p40)*np.exp(2*1j*np.pi*p52) + np.exp(-2*1j*np.pi*p39)*np.exp(2*1j*np.pi*p51) + np.exp(-2*1j*np.pi*p38)*np.exp(2*1j*np.pi*p50) + np.exp(-2*1j*np.pi*p37)*np.exp(2*1j*np.pi*p49) + np.exp(-2*1j*np.pi*p36)*np.exp(2*1j*np.pi*p48)





















    # for i in range(M.shape[0]):
    #     # Handle the first element of each row
    #     tuples.append((M[i][0], i + 1))
    #     tuples.append((M[i][0], i + 1 + 3 * increment))
    #     # Special case for the middle index
    #     tuples.append((0, i + 1 + increment))
    #     # next pair
    #     tuples.append((M[i][1], i + 1 + 2 * increment))
    #     tuples.append((M[i][1], i + 1 + 4 * increment))

    #     if M.shape[1] > 2:
    #         # Handle subsequent elements of each row
    #         for j in range(2,M.shape[1]):
    #             int_j = j + 2
    #             index1 = i + 1 + increment + int_j * increment
    #             index2 = index1 + 2*increment
    #             # check if index2 is out of bounds
    #             if index2 > d**2//2 - d//2:
    #                 index2 -=  increment

    #             tuples.append((M[i][j], index1))
    #             tuples.append((M[i][j], index2))









    # now add the phase
    # row a given row, 0 to 2 pi in d/2 steps
    # for i in range(1, d//2):
    #     print(f'row {i}')
    #     print(numerical_params)
    #     numerical_params_i = []
    #     # numerical_params_i = [((j+i)/(d/2), i+d//2*j) for j in range(d-1)]
    #     for j in range(0, d//2-1)
    #     base_phase_ls = [(k*(j+i))/(d//2) for k in range(1,d//2)]
    #     # add manually the first two
    #     numerical_params_i.append((base_phase_ls[0], i))
    #     numerical_params_i.append((0, i+d//2))
    #     numerical_params_i.append((base_phase_ls[0], i+d//2*(3)))
    #     # repeat each phase twice
    #     for j in range(1, len(base_phase_ls)):
    #         index = i+d//2*(j+1)
    #         numerical_params_i.append((base_phase_ls[j], i+d//2*(j+1)))
    #         numerical_params_i.append((base_phase_ls[j], index+2*d//2))
                
    #     numerical_params += numerical_params_i

    # for i in range

    # cehck that the phases are orthogonal
    # get_inner_prods(d, numerical_params=numerical_params)
    # print(numerical_params)

    # i = 2
    # d = 10
    # # print([k*i/(d//2) for k in range(1,d//2+1)])
    # print('----------')
    # # for j in range(0, d//2-1):
    # #     print([(k*(j+i))/(d//2) for k in range(1,d//2)])
    # #     print([np.exp(2*np.pi*1j*(k*(i+j))/(d//2)) for k in range(1,d//2)])
    # #     print('\n')
    # print([np.exp(2*np.pi*1j*k)/(d//2)) for k in range(1,d//2)])

    # def create_orthogonal_vectors(d):
    #     '''Creates orthogonal vectors for a given dimension d'''
    #     n = np.arange(d/2)
    #     k = n.reshape((d//2, 1))
    #     omega = np.exp(-2j * np.pi * k * n / (d/2))

    #     return omega

    # d = 10  # Example size
    # vectors = create_orthogonal_vectors(d)
    # print(vectors)



    # numerical_params+=[(1/3, 1), (2/3, 7), (1/3, 4), (2/3, 10), (0, 13), (1/3, 2), (1/3, 5), (2/3, 8), (2/3, 11), (0, 14)]

    # print(numerical_params)

    # best = [0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3]
    # best_params = [(best[i], i) for i in range(len(best))]

    # p = 5
    # print(sum([np.exp(2*np.pi*1j*p_v*(j)/(d//2)) for j in range(1,d//2)]))
    # for p_v in range(1, d//2):
    #     print([np.exp(2*np.pi*1j*p_v*(j)/(d//2)) for j in range(1,d//2)])



    # find inner products
    # get_inner_prods(d, numerical_params=best_params)
    # print(sum([np.exp(2*np.pi*1j*(j+1)/(d/2)) for j in range(d-1)]))

    # print(sum([np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3), np.exp(4*np.pi*1j/3), 1]))

    # for j in range(d-1):
    #     for l in range(d-1):
    #         numerical_params.append((base_ls[l], j+(d//2)*l))
    # print(numerical_params)
    # print(len(numerical_params))


    # simplify_system(d, numerical_params=numerical_params)





    
    # imagine creating params as matrix where each row represents p_j, p_{j + d/2}, p_{j + d}, ...
    # each column will have the same

        


    # show_orthogonality(d)
    # get_det([0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3])

    # print(sum([np.exp(2*np.pi*1j/3), 1, np.exp(4*np.pi*1j/3), np.exp(8*np.pi*1j/3), np.exp(4*np.pi*1j/3)]))
    # print(sum([np.exp(2*np.pi/1j*i/4) for i in range(5)]))



    # get evenly spaced phase params from 0 to 2pi in d^2/2 - d/2 steps
    # base = np.linspace(0, 2*np.pi, num=(d//2))
    # # get d-2 random permutations of the base
    # params = list(np.random.permutation(base))
    # what if set diagonal to be 0 to 2pi in d-1 steps, rest 0?
    # base = np.linspace((d-1)/(2*np.pi), 2*np.pi, num=(d-1+2))
    # # create all 0 vector for params
    # params = [0 for _ in range(d**2//2-d//2)]
    # print(len(params))
    # # set diagonal to be base
    # for i in range(d-1):
    #     print((i+1)*d//2)
    #     params[(i+1)*d//2] = base[i]
    # indices = [0, 4, 6, 7, 11, 12, 14] # for d = 6
    # # indices = [0, 3, 4, 5]
    # for i in indices:
    #     params[i] = base[indices.index(i)]

    # # print(params)
    # # get_inner_prod_sub(d, params, num_vecs=2)
    # get_det(params)


    # for _ in range(d-2):
    #     params += list(np.random.permutation(base))
    # print(params)
    # get_inner_prod_sub(d, params)

    # best_params =[0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3]
    # get_det(best_params)
    # get_inner_prod_sub(d, best_params)





    # print(get_det(params))
    # print(params)
    # print(get_vectors(params))
    # print(loss_phase(params, d, guess=False))
    # params = [0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3]
    # get_det(params)


    ## old d = 10 numerical results
    #loss_num2 = 1.121236366580341e-05
    # params_num2 = [0.49999997271298846, -0.8356697046226463, -0.6901289076337005, -0.9331429494907373, -0.16206457394431045, -0.5000000367240022, -0.43566973741722875, -0.890128957757636, 0.26685699711166727, -1.5620646086412582, -0.5000000378797359, -2.235669731293962, -0.49012895748565904, -1.1331430010278547, 0.23793539481588633, 0.9999999616216863, -1.400000096962391, 0.3999998998223931, -0.20000010275611058, -0.8000001073516739, 0.9999999596356349, -0.2000000936732211, 0.19999990198821, -0.6000000953637014, 0.599999899801483, -4.174821191195496e-08, -0.8000000996216404, 0.7999998935621904, -0.40000009449924656, -0.6000000966388638, -0.5000000421796911, 1.3643302836362694, -1.2901289187555183, -0.3331429622012385, 0.6379354079374747, -4.0704953106727827e-08, -0.600000083086426, 0.5999999152591754, -0.8000000963266489, 0.7999999018863323, -0.500000034486441, -1.0356697325788677, -0.09012894955181719, -1.53314299475126, 0.037935392650653856]

    # # has loss 1.121236366580341e-05
    # params_ana2 = [(1/2,0), -0.8356697046226463, -0.6901289076337005, -0.9331429494907373, -0.16206457394431045, (-1/2, 5), -0.43566973741722875, -0.890128957757636, 0.26685699711166727, -1.5620646086412582, (-1/2, 10), -2.235669731293962, -0.49012895748565904, -1.1331430010278547, 0.23793539481588633, (1, 15), (-1.4, 16), (0.4, 17), (-0.2, 18), (-0.8, 19), (1, 20), (-0.2, 21), (0.2, 22), (-0.6, 23), (0.6, 24), (0, 25), (-0.8, 26), (0.8, 27), (-0.4, 28), (-0.6, 29), (-1/2, 30), 1.3643302836362694, -1.2901289187555183, -0.3331429622012385, 0.6379354079374747, (0, 35), (-0.6, 36), (0.6, 37), (-0.8, 38), (0.8, 39), (-1/2, 40), -1.0356697325788677, -0.09012894955181719, -1.53314299475126, 0.037935392650653856]

