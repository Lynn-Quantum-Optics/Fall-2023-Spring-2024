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

def loss_phase(params, d, guess, replacement_params=None):
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

    rref_matrix, pivot_columns = result_matrix.rref()

    # rref_matrix is the row-reduced echelon form of your matrix
    print("Row-reduced echelon form:")
    print(rref_matrix)

    # pivot_columns are the indices of the pivot columns
    print("Pivot columns:", pivot_columns)

    
if __name__ == '__main__':
    # loss_num1= 3.518236836123831e-05
    # params_num1= [-0.5000003021751246, 0.8297261207122422, 0.4002740820670201, -0.24187654563870226, -0.036890852462187464, 0.49999969026391267, 0.6297261226700273, -0.39972592038372085, -0.8418765441599031, 0.5631091443171757, -0.5000002990545374, 0.2297261175948017, 0.00027407425516466307, -1.0418765582144849, 0.7631091378953769, -3.1341316201935654e-07, -0.6000001279508675, 0.19999974006658813, 0.7999997549735715, -0.4000001712524025, -0.5000003039751297, -0.9702738700872592, 1.2002740911151337, -0.641876543160646, 0.3631091477758074, -3.2204363112297466e-07, 0.5999998095181849, -0.20000030392990611, 1.1999997075787876, -0.600000233255531, -3.0635065754545956e-07, -0.20000021245432606, -0.6000003197638013, 1.5999996970758865, -0.8000002532317712, -0.5000003180771708, 0.4297261220252235, 0.8002740773310877, -1.441876550125145, -0.8368908584033007, -3.1535010174187685e-07, -0.8000001903296595, 0.5999997282862437, 0.3999997403344088, -0.2000002332089973]
    # params_ana1 = [-1/2, 0.8297261207122422,0.4, -0.24187654563870226, -0.036890852462187464, 1/2, 0.6297261226700273, -0.4, -0.8418765441599031, 0.5631091443171757, -1/2, 0.2297261175948017, 0.00027407425516466307, -1.0418765582144849, 0.7631091378953769, 0, -0.6, 0.2, 0.8, -0.4, -1/2, -0.9702738700872592, 1.2, -0.641876543160646, 0.3631091477758074, 0, 0.6, -0.2, 1.2, -0.6, 0, -0.2, -0.6, 1.6, -0.8, -0.5, 0.4297261220252235, 0.8, -1.441876550125145, -0.8368908584033007, 0, -0.8, 0.6, 0.4, -0.2]
    loss_num2 = 1.121236366580341e-05
    params_num2 = [0.49999997271298846, -0.8356697046226463, -0.6901289076337005, -0.9331429494907373, -0.16206457394431045, -0.5000000367240022, -0.43566973741722875, -0.890128957757636, 0.26685699711166727, -1.5620646086412582, -0.5000000378797359, -2.235669731293962, -0.49012895748565904, -1.1331430010278547, 0.23793539481588633, 0.9999999616216863, -1.400000096962391, 0.3999998998223931, -0.20000010275611058, -0.8000001073516739, 0.9999999596356349, -0.2000000936732211, 0.19999990198821, -0.6000000953637014, 0.599999899801483, -4.174821191195496e-08, -0.8000000996216404, 0.7999998935621904, -0.40000009449924656, -0.6000000966388638, -0.5000000421796911, 1.3643302836362694, -1.2901289187555183, -0.3331429622012385, 0.6379354079374747, -4.0704953106727827e-08, -0.600000083086426, 0.5999999152591754, -0.8000000963266489, 0.7999999018863323, -0.500000034486441, -1.0356697325788677, -0.09012894955181719, -1.53314299475126, 0.037935392650653856]

    # has loss 1.121236366580341e-05
    params_ana2 = [(1/2,0), -0.8356697046226463, -0.6901289076337005, -0.9331429494907373, -0.16206457394431045, (-1/2, 5), -0.43566973741722875, -0.890128957757636, 0.26685699711166727, -1.5620646086412582, (-1/2, 10), -2.235669731293962, -0.49012895748565904, -1.1331430010278547, 0.23793539481588633, (1, 15), (-1.4, 16), (0.4, 17), (-0.2, 18), (-0.8, 19), (1, 20), (-0.2, 21), (0.2, 22), (-0.6, 23), (0.6, 24), (0, 25), (-0.8, 26), (0.8, 27), (-0.4, 28), (-0.6, 29), (-1/2, 30), 1.3643302836362694, -1.2901289187555183, -0.3331429622012385, 0.6379354079374747, (0, 35), (-0.6, 36), (0.6, 37), (-0.8, 38), (0.8, 39), (-1/2, 40), -1.0356697325788677, -0.09012894955181719, -1.53314299475126, 0.037935392650653856]

    # print(loss_phase(params_num2, 10, guess=False, replacement_params=params_ana2))


    # get_inner_prods(10, numerical_params=params_ana2, apply_guess=False, solve=True)
    # (1/2, 6), (0, 2), (1/2, 7), (0, 3), (1/2, 8), (0, 4), (0, 9), (0, 31), (1/2, 41), (0, 32), (1/2, 42), (0, 33), (0, 43), (0, 34), (1/2, 44)
    params_ana3 = params_ana2 + [(0,1), (0,2)]
    simplify_system(10, params_ana3)


    # row_reduce(6)

