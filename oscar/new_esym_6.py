# file to test new "symmetrized" bell basis
import numpy as np
import itertools
from oscars_toolbox.trabbit import trabbit

## helper functions for generate ideas for vectors ##
def get_comb(total_vec, num_neg):
    ''' generate all combinations of positions for num_neg -1 in a vector of length total_vec'''
    positions = range(total_vec)  # Positions in the vector (0 to 5)
    combinations = list(itertools.combinations(positions, num_neg))

    # generate the vectors based on the combinations
    vectors = []
    for combo in combinations:
        vector = [1]*total_vec  # start with all 1s
        for pos in combo:
            vector[pos] = -1  # set -1s at the specified positions
        vectors.append(vector)
        print(vector)
    return vectors

def structured_gram_schmidt(vectors):
    '''BROKEN'''
    orthogonal_set = []
    for v in vectors:
        for u in orthogonal_set:
            projection = np.dot(u, v) / np.dot(u, u) * u
            # Adjust the projection to maintain the structure
            for i in range(0, len(projection), 2):
                avg = (projection[i] - projection[i + 1]) / 2
                projection[i], projection[i + 1] = avg, -avg
            v -= projection
        orthogonal_set.append(v)
    return orthogonal_set

def get_old_phase(d=3):
    '''Get the phase vectors for d = 3 as in the exisiting definition'''
    return [[np.exp(2 * np.pi * 1j * i * p / d) for i in range(d)] for p in range(d)]

## helper functions to confirm whether our basis idea is valid # 
def check_orthogonal(vec_ls):
    # orthogonal = True
    for i in range(len(vec_ls)):
        for j in range(len(vec_ls)):
            if i != j:
                if not(np.isclose(np.abs(np.array(vec_ls[i]) @ np.array(vec_ls[j]).conj().T), 0, atol=1e-12)):
                    print(f'vector {i} and {j} are not orthogonal and have dot product {np.abs(np.array(vec_ls[i]) @ np.array(vec_ls[j]).conj().T)}')
                    # orthogonal = False
                    return False
    print('all vectors are orthogonal.')
    return True

def check_constraints_and_orthogonality(vectors):
    '''Checks if the basis is eigenstate of particle exchange'''
    # Check orthogonality
    orthogonality = check_orthogonal(vec_ls=vectors)

    # Check the structural constraints: no zero entries and pairing
    no_zero_entries = not np.any(vectors == 0)

    correct_pairing = True
    for v in vectors:
        scale = v[1] / v[0]
        for i in range(2, len(v), 2):
            scale_i = v[i+1] / v[i]
            if not np.isclose(scale, scale_i, atol=1e-33):
                print(f'v: {v}')
                print(f'scale: {scale}')
                print(f'scale_i: {scale_i}')
                correct_pairing = False
                break

    print(f'orthogonality: {orthogonality}')
    print(f'no_zero_entries: {no_zero_entries}')
    print(f'correct_pairing: {correct_pairing}')

    return orthogonality and no_zero_entries and correct_pairing

## learning phase ##
def get_vectors_nonunity(params):
    '''Convert the learned params into phase vectors'''
    # unpack the params
    real_ls = params[:len(params)//2]
    imag_ls = params[len(params)//2:]

    vectors = [
        np.array([1, 1, 1, 1, 1, 1], dtype=complex),
        np.array([real_ls[1] + 1j*imag_ls[1], (real_ls[0] + 1j*imag_ls[0])*(real_ls[1] + 1j*imag_ls[1]), real_ls[2] + 1j*imag_ls[2], (real_ls[0] + 1j*imag_ls[0])*(real_ls[2] + 1j*imag_ls[2]), real_ls[3]+1j*imag_ls[3], (real_ls[0] + 1j*imag_ls[0])*(real_ls[3] + 1j*imag_ls[3])], dtype=complex),
        np.array([real_ls[5] + 1j*imag_ls[5], (real_ls[4] + 1j*imag_ls[4])*(real_ls[5] + 1j*imag_ls[5]), real_ls[6] + 1j*imag_ls[6], (real_ls[4] + 1j*imag_ls[4])*(real_ls[6] + 1j*imag_ls[6]), real_ls[7]+1j*imag_ls[7], (real_ls[4] + 1j*imag_ls[4])*(real_ls[7] + 1j*imag_ls[7])], dtype=complex),
        np.array([real_ls[9] + 1j*imag_ls[9], (real_ls[8] + 1j*imag_ls[8])*(real_ls[9] + 1j*imag_ls[9]), real_ls[10] + 1j*imag_ls[10], (real_ls[8] + 1j*imag_ls[8])*(real_ls[10] + 1j*imag_ls[10]), real_ls[11]+1j*imag_ls[11], (real_ls[8] + 1j*imag_ls[8])*(real_ls[11] + 1j*imag_ls[11])], dtype=complex),
        np.array([real_ls[13] + 1j*imag_ls[13], (real_ls[12] + 1j*imag_ls[12])*(real_ls[13] + 1j*imag_ls[13]), real_ls[14] + 1j*imag_ls[14], (real_ls[12] + 1j*imag_ls[12])*(real_ls[14] + 1j*imag_ls[14]), real_ls[15]+1j*imag_ls[15], (real_ls[12] + 1j*imag_ls[12])*(real_ls[15] + 1j*imag_ls[15])], dtype=complex),
        np.array([real_ls[17] + 1j*imag_ls[17], (real_ls[16] + 1j*imag_ls[16])*(real_ls[17] + 1j*imag_ls[17]), real_ls[18] + 1j*imag_ls[18], (real_ls[16] + 1j*imag_ls[16])*(real_ls[18] + 1j*imag_ls[18]), real_ls[19]+1j*imag_ls[19], (real_ls[16] + 1j*imag_ls[16])*(real_ls[19] + 1j*imag_ls[19])], dtype=complex),
        ]
    
    # normalize
    for i in range(len(vectors)):
        vectors[i] /= np.linalg.norm(vectors[i])

    # print('comparison')
    # print(vectors[1][0] / vectors[1][1])
    # print(vectors[1][2] / vectors[1][3])
    
    return vectors

def get_vectors(params):
    '''Assumes each entry is some root of unity'''

    vectors = [
                    np.array([1,1,1,1,1,1], dtype=complex),
                    np.array([1, np.exp(2*np.pi*1j*params[0]), np.exp(2*np.pi*1j*params[1]), np.exp(2*np.pi*1j*(params[0]+params[1])), np.exp(2*np.pi*1j*params[2]), np.exp(2*np.pi*1j*(params[0]+params[2]))], dtype=complex),
                    np.array([1, np.exp(2*np.pi*1j*params[3]), np.exp(2*np.pi*1j*params[4]), np.exp(2*np.pi*1j*(params[3]+params[4])), np.exp(2*np.pi*1j*params[5]), np.exp(2*np.pi*1j*(params[3]+params[5]))], dtype=complex),
                    np.array([1, np.exp(2*np.pi*1j*params[6]), np.exp(2*np.pi*1j*params[7]), np.exp(2*np.pi*1j*(params[6]+params[7])), np.exp(2*np.pi*1j*params[8]), np.exp(2*np.pi*1j*(params[6]+params[8]))], dtype=complex),
                    np.array([1, np.exp(2*np.pi*1j*params[9]), np.exp(2*np.pi*1j*params[10]), np.exp(2*np.pi*1j*(params[9]+params[10])), np.exp(2*np.pi*1j*params[11]), np.exp(2*np.pi*1j*(params[9]+params[11]))], dtype=complex),
                    np.array([1, np.exp(2*np.pi*1j*params[12]), np.exp(2*np.pi*1j*params[13]), np.exp(2*np.pi*1j*(params[12]+params[13])), np.exp(2*np.pi*1j*params[14]), np.exp(2*np.pi*1j*(params[12]+params[14]))], dtype=complex),
               ]
    return vectors

def loss_phase(params):
    '''Loss function to find the optimal phases for d = 6'''

    vectors = get_vectors(params)

    # now find inner products
    inner_products = 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            inner_products += np.abs(np.dot(vectors[i], vectors[j].conj().T))

    return inner_products

def random_gen(num_params=15):
    '''Generate random parameters'''
    return np.random.uniform(-1, 1, size=(num_params))
    
def optimize_phase(tol=1e-10, x0=None):
    '''Minimize the loss function of sum of abs value of inner products to find the optimal phases'''
    if x0 is None:
        x_best, loss_best = trabbit(loss_func=loss_phase, random_gen=random_gen, alpha=1, tol=tol, temperature=0.01)
    else:
        x_best, loss_best = trabbit(loss_func=loss_phase, random_gen=random_gen, alpha=1, tol=tol, temperature=0.01, x0_ls=[x0])
    print(f'best loss: {loss_best}')
    print(f'best params: {list(x_best)}')
    return x_best, loss_best

## OLD attempts to numerically solve for the remaining parameters. ##

def loss_phase_remaining(params):
    if np.isclose(params[1], 0, atol=1e-10):
        params[1] == 1
    if np.isclose(params[3], 0, atol=1e-10):
        params[3] == 1
    if np.isclose(params[5], 0, atol=1e-10):
        params[5] == 1
    if np.isclose(params[7], 0, atol=1e-10):
        params[7] == 1
    param0 = np.exp(2*np.pi*1j*np.sqrt(int(params[0])/int(params[1])))
    param1 = np.exp(2*np.pi*1j*np.sqrt(int(params[2])/int(params[3])))
    param2 = np.exp(2*np.pi*1j*np.sqrt(int(params[4])/int(params[5])))
    param3 = np.exp(2*np.pi*1j*np.sqrt(int(params[6])/int(params[7])))
    params_tot = [param0, 1/3, -1/3, 0.5, -1, 0,param1, 2/3, 1/3, param2, 4/3, -1/3, param3, 2/3, 1/3]
    return loss_phase(params_tot)

def random_remaining_gen(num_params=8):
    '''Generate random parameters'''
    return np.random.uniform(1, 10, size=(num_params))

def optimize_remaining(tol=1e-10):
    x_best, loss_best = trabbit(loss_func=loss_phase_remaining, random_gen=random_remaining_gen, alpha=1, tol=tol, temperature=0.01)
    print(f'best loss: {loss_best}')
    print(f'best params: {list(x_best)}')
    return x_best, loss_best

## sympy code ##
import sympy as sp

def custom_chop(expr, tol=1e-15):
    '''Removes small values (below tol) in sympy expressions in order to simplify numerical + symbolic expressions'''
    if expr is None:
        print("Encountered None expression")
        return None
    elif expr.is_Number:
        if abs(expr) < tol:
            return sp.Integer(0)
        else:
            return expr
    elif expr.is_Symbol:
        return expr
    else:
        if not expr.args:  # If expr.args is empty
            return expr
        chopped_args = [custom_chop(arg, tol) for arg in expr.args]
        if None in chopped_args:
            print("None found in arguments of:", expr)
            return None
        return expr.func(*chopped_args)

def get_inner_prods(numerical_params=None, solve=False):
    '''Compute symbolic inner products between vectors, either completely symbolically or with numerical values for some of the parameters.

    Params:
        numerical_params: list of numerical values for some of the parameters. If None, then purely symbolic params used.
        solve: whether to solve analytically for the remaining parameters (if numerical_params is not None). NOTE: this did not ever finish running, so I wouldn't use it but am leaving it here for reference.

    Returns:
        results: symbolic matrix of inner products between vectors

    '''

    # start indexing from 0
    params = sp.symbols('p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14', real=True)

    vectors = [
        sp.Matrix([1, 1, 1, 1, 1, 1]),
        sp.Matrix([1, sp.exp(2*sp.pi*sp.I*params[0]), sp.exp(2*sp.pi*sp.I*params[1]), sp.exp(2*sp.pi*sp.I*(params[0]+params[1])), sp.exp(2*sp.pi*sp.I*params[2]), sp.exp(2*sp.pi*sp.I*(params[0]+params[2]))]),
        sp.Matrix([1, sp.exp(2*sp.pi*sp.I*params[3]), sp.exp(2*sp.pi*sp.I*params[4]), sp.exp(2*sp.pi*sp.I*(params[3]+params[4])), sp.exp(2*sp.pi*sp.I*params[5]), sp.exp(2*sp.pi*sp.I*(params[3]+params[5]))]),
        sp.Matrix([1, sp.exp(2*sp.pi*sp.I*params[6]), sp.exp(2*sp.pi*sp.I*params[7]), sp.exp(2*sp.pi*sp.I*(params[6]+params[7])), sp.exp(2*sp.pi*sp.I*params[8]), sp.exp(2*sp.pi*sp.I*(params[6]+params[8]))]),
        sp.Matrix([1, sp.exp(2*sp.pi*sp.I*params[9]), sp.exp(2*sp.pi*sp.I*params[10]), sp.exp(2*sp.pi*sp.I*(params[9]+params[10])), sp.exp(2*sp.pi*sp.I*params[11]), sp.exp(2*sp.pi*sp.I*(params[9]+params[11]))]),
        sp.Matrix([1, sp.exp(2*sp.pi*sp.I*params[12]), sp.exp(2*sp.pi*sp.I*params[13]), sp.exp(2*sp.pi*sp.I*(params[12]+params[13])), sp.exp(2*sp.pi*sp.I*params[14]), sp.exp(2*sp.pi*sp.I*(params[12]+params[14]))]),
    ]
    
    n = len(vectors)
    results = sp.Matrix.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            results[i, j] = vectors[i].dot(vectors[j].conjugate().T)
            results[j, i] = results[i, j].conjugate()

    # if numerical values are not given, then just print out the expressions
    if numerical_params is None:
        for i in range(n):
            print('\Vec{I}'+f'_{i}'+ ' = \\begin{bmatrix}')
            for j in range(n):
                expr = results[i, j]
                expr = sp.simplify(expr)
                sp.print_latex(expr)
                print('\\\\')
            print('\\end{bmatrix},\\\\')

    else: # substitute in the values we have
         # set specific numerical values to some of the params
        test_params = []
        for i in range(len(params)):
            if i in [0, 6, 9, 12]:
                test_params.append(params[i])
            else:
                test_params.append(numerical_params[i])

        # apply to results
        results = results.subs({params[i]: test_params[i] for i in range(len(params))})

        for i in range(n):
            print('\Vec{I}'+f'_{i}'+ ' = \\begin{bmatrix}')
            for j in range(n):
                expr = results[i, j]
                expr = sp.N(expr)
                expr = custom_chop(expr)
                expr = sp.simplify(expr)
                sp.print_latex(expr)
                print('\\\\')
            print('\\end{bmatrix},\\\\')

    if solve and numerical_params is not None:
        # solve for values that make sum of abs value of inner products = 0
        results = results.subs({params[i]: numerical_params[i] for i in range(len(params))})
        # print out columns
        sp.pprint(sp.N(results))

        # solve for values that make sum of abs value of inner products = 0
        total_sum = 0
        for i in range(n):
            for j in range(n):
                total_sum += sp.Abs(results[i, j])

        # input test params into total_sum
        print('total_sum: ')
        total_sum = total_sum.subs({params[i]: test_params[i] for i in range(len(params))})

        remaining_params = sp.symbols('p0 p6 p9 p12', real=True)

        # Solve the equation (if it's solvable)
        solution = sp.nsolve(total_sum,remaining_params )

        print(f'solution: {solution}')

    
    return results    

if __name__ == '__main__':

    ## solution obtained through optimize_phase() ##

    # best_loss_numerical = 8.346005000999512e-07
    # best_params_numerical = [0.3073524627378583, 0.3333333179399618, -0.33333334259706254, 0.4999999920193058, -1.0000000130072773, -1.2082227449358706e-08, -0.02563931056721057, 0.6666666524787896, 0.33333331987509573, 0.8073524630069947, 1.3333333168849828, -0.33333334966782724, -0.5256393104295811, 0.666666651145216, 0.3333333221305492]
    
    # loss after substituting in for all but p0, p6, p9. p12
    # best_loss = 7.667367120048474e-09
    # best_params_numerical = [0.3073524627378583, 0.3333333179399618, -0.33333334259706254, 0.4999999920193058, -1.0000000130072773, -1.2082227449358706e-08, -0.02563931056721057, 0.6666666524787896, 0.33333331987509573, 0.8073524630069947, 1.3333333168849828, -0.33333334966782724, -0.5256393104295811, 0.666666651145216, 0.3333333221305492]
    best_params_approx = [0.3073524627378583, 1/3, -1/3, 1/2, -1, 0, -0.02563931056721057, 2/3, 1/3, 0.8073524630069947, 4/3, -1/3, -0.5256393104295811, 2/3, 1/3]

    get_inner_prods(best_params_approx)

    # actual best params determined with get_inner_prods() 
    # best_params = [0, 1/3, -1/3, 1/2, -1, 0, 0, 2/3, 1/3, 1/2, 4/3, -1/3, 1/2, 2/3, 1/3]
    

    # vectors = get_vectors(best_params)
    # for i in range(len(vectors)):
    #     print(f'vectors[{i}]: {vectors[i]}')
    # check_constraints_and_orthogonality(vectors)
