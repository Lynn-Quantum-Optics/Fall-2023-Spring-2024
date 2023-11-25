# file to find a linear combiantion of pauli and gell man operators that yields a U in the single particle basis
# currently only works for d = 2^n or d = 3^n or d = 2*n*3^m

import numpy as np
import sympy as sp
import scipy
from functools import partial
from scipy.optimize import shgo, basinhopping
from math import comb
from tqdm import trange
import numpy as np
import scipy.linalg
from bell import make_bell, CNS, num_overlapping

## ----------- parametrizing SU(n) ----------- #
# helper function to check if valid matrix #
def is_hermitian(matrix, tol=1e-10):
    '''Checks if a matrix is Hermitian.'''
    return np.allclose(matrix, matrix.conj().T, atol=tol)

def is_traceless(matrix, tol=1e-10):
    '''Checks if a matrix is traceless.'''
    return np.isclose(np.trace(matrix), 0, atol=tol)

def is_unitary(matrix, tol=1e-10):
    '''Checks if a matrix is unitary.'''
    return np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=tol)

def is_param_valid(n):
    '''Checks if a the param process yields a valid SU(n) matrix by checking the generator construction and unitarity of final product and det 1'''
    
    # check if the matrix is hermitian and traceless
    generators = get_su_n_generators(n)
    for i, g in enumerate(generators):
        assert is_hermitian(g), f'Generator {i} is not Hermitian'
        assert is_traceless(g), f'Generator {i} is not traceless'

    # generate a random SU(n) matrix
    params = np.random.random(len(generators))
    U = get_su_n(params, generators)

    # check if U is unitary
    assert is_unitary(U), 'U is not unitary'
    assert is_det1(U), 'U does not have determinant 1'

    return is_unitary(U) and is_det1(U)

def is_valid_U_n(U):
    '''Confirms that matrix U'''
    return is_unitary(U)

def get_u_n_generators(n):
    '''Returns the generators of u(n) as a list of numpy arrays. By definition, need A^dagger = -A and A^dagger = A^T for A in generators.
    
    '''
    generators = []
    # add n-1 diagonal generators
    for i in range(n):
        diag_matrix = np.zeros((n, n), dtype=complex)
        diag_matrix[i, i] = -1
        generators.append(diag_matrix)

    # add off-diagonal generators
    for i in range(n):
        for j in range(i + 1, n):
            real_matrix = np.zeros((n, n), dtype=complex)
            real_matrix[i, j] = real_matrix[j, i] = 1
            generators.append(real_matrix)

            imag_matrix = np.zeros((n, n), dtype=complex)
            imag_matrix[i, j] = -1j
            imag_matrix[j, i] = 1j
            generators.append(imag_matrix)

    # print('Number of generators:', len(generators))
    return generators

def get_U_n(params, generators):
    '''Exponentiates a linear combination of the generators of u(n).'''
    params = np.array(params)
    generator_sum = sum(p * g for p, g in zip(params, generators))
    # print(type(params[0])=='numpy.float64')
    # if type(params[0]) == 'numpy.float64':
    mat =  scipy.linalg.expm(1j*generator_sum)
    assert is_valid_U_n(mat), 'U is not unitary'
    return mat
    # else:
        # return sp.Matrix(1j*generator_sum).exp()

def find_params(d, numerical=True):
    '''finds the optimal params to generate the unitary matrix.

    Params:
        d (int): dimension of the unitary matrix
        numerical (bool): whether to use numerical optimization or sympy
    '''
    # get the function to get the unitary matrix, 2d x 2d
    generators = get_u_n_generators(2*d)
    U = partial(get_U_n, generators=generators)
    num_params = len(generators)*2

    # starting at k = d, find largest group of k bell states such that when we apply U^{-1} \otimes U^{-1} to the bell states, we get non overlapping detection signatures
    # we start with a particular k and see if we can find a group of k bell states that work; if so, increase k by 1 and repeat
    # if not, we have found the largest group of bell states that works
    def N(param_vals, k, i):
        '''returns the number of overlapping detection signatures for a given index of C(d^2, k) bell states'''
        # get the k-tuple
        k_tuple = CNS(d**2, k, i)
        bell_states = np.zeros((4*d**2, k), dtype=complex)
        j = 0
        for state_num in k_tuple:
            # determine the correlation class and phase class
            c = state_num // d
            p = state_num % d
            # get the bell state and append as a column vector
            bell_states[:,j] = make_bell(d, c, p, b=True, options='none', joint=False).reshape(4*d**2)
            # print('(c, p)', (c, p))
            # print(bell_states[:, j])
            j+=1
        # print('bell_states', bell_states)
        # apply U  = \mathcal{U}^{-1} \otimes \mathcal{U}^{-1} to the bell states
        # print(type(param_vals[0]))
        U_inv1 = U(param_vals[:len(param_vals)//2]).conj().T
        U_inv2 = U(param_vals[len(param_vals)//2:]).conj().T
        U_tot = np.kron(U_inv1, U_inv2)
        bell_proj = U_tot @ bell_states
        return num_overlapping(bell_proj)

    def prove_k(k):
        '''returns True if we can find k bell states that work, and False otherwise'''
        # get all combinations of k bell states via CNS
        # for each combination, apply U^{-1} \otimes U^{-1} to the bell states and check if the detection signatures overlap
        # if they do, return False
        # if we get through all combinations without returning False, return True

        for i in trange(comb(d**2, k)):
            if numerical:
           # solve for the param_vals that make N = 0
                x0 = [0 for _ in range(num_params)]
                # altnerate 0 and 1
                # x0 = [0 if i % 2 == 0 else 1 for i in range(num_params)]
                print('initial guess', x0)
                init_N = N(x0, k, i)
                print('initial N', init_N)
                result = init_N
                if np.isclose(result, 0, atol=1e-10):
                    return True
                # define a callback function to print the progress of the optimization
                basinhopping
                result = basinhopping(func=N, x0=x0, minimizer_kwargs={"args": (k, i)}, niter=10, disp=False)
                # call shgo to find the minimum
                # result = shgo(func=N, args = (k, i), bounds = [(-1, 1) for _ in range(num_params)])
                print(U(result.x))
                print(U(result.x).shape)
                print('min result', result.x)
                print(result.fun)
                if np.isclose(result.fun, 1e-10):
                    return True
            else:
                # solve for the param_vals that make N = 0
                param_vals = sp.symbols('p0:%d' % num_params)
                result = sp.nsolve(N(param_vals, k, i), param_vals, [0 for _ in range(num_params)])
                print(result)
                if np.isclose(result, 0, atol=1e-10):
                    return True

        # if we get through all combinations without returning False, return False
        return False             

    # get all combinations of k bell states
    k = d
    print(f'Finding optimal k.... Try k = {k} first')
    while prove_k(k):
        print(f'{k} works!')
        k += 1

if __name__ == '__main__':
    find_params(2, numerical=True)


    

   


    
