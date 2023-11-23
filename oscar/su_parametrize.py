# file to find a linear combiantion of pauli and gell man operators that yields a U in the single particle basis
# currently only works for d = 2^n or d = 3^n or d = 2*n*3^m

import numpy as np
import sympy as sp
import scipy
from scipy.optimize import shgo, basinhopping
from math import comb
from tqdm import trange
from bell import make_bell, CNS, num_overlapping

# define the pauli matrices
Sx = np.array([[0, 1], [1, 0]], dtype = complex)
Sy = np.array([[0, -1j], [1j, 0]], dtype = complex)
Sz = np.array([[1, 0], [0, -1]], dtype = complex)
I2 = np.eye(2, dtype = complex)
# define gell mann matrices
L1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype = complex)
L2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype = complex)
L3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype = complex)
L4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype = complex)
L5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype = complex)
L6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype = complex)
L7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype = complex)
L8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype = complex) / np.sqrt(3)
# define the identity matrix
I3 = np.eye(3, dtype = complex)

def su_2(alpha_1, alpha_2, alpha_3):
    '''expresses a 2x2 unitary matrix as a linear combination of the pauli matrices'''
    # alpha_1, alpha_2, alpha_3 are the coefficients of the pauli matrices and are real
    assert np.isreal(alpha_1) and np.isreal(alpha_2) and np.isreal(alpha_3)
    anti_hermitian = 1j * (alpha_1*Sx + alpha_2*Sy + alpha_3*Sz)

    # Exponential map to SU(2)
    return scipy.linalg.expm(anti_hermitian)

def su_3(beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8):
    '''expresses a 3x3 unitary matrix as a linear combination of the gell mann matrices'''
    # beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8 are the coefficients of the gell mann matrices and are real
    assert np.isreal(beta_1) and np.isreal(beta_2) and np.isreal(beta_3) and np.isreal(beta_4) and np.isreal(beta_5) and np.isreal(beta_6) and np.isreal(beta_7) and np.isreal(beta_8)
    # need to normalize the coefficients
    # find the norm of the vector (beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8)
    n = np.array([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8])
    norm = np.linalg.norm(n)
    # normalize the vector
    n /= norm
    return scipy.linalg.expm(1j*(n[0]*L1 + n[1]*L2 + n[2]*L3 + n[3]*L4 + n[4]*L5 + n[5]*L6 + n[6]*L7 + n[7]*L8))

def get_U(d):
    '''returns the function to get U parametrized by the coefficients of the pauli and gell mann matrices'''
    # fitting for a 4*d^2 unitary matrix in single
    # factor 4*d^2 into 2^n * 3^m
    # find n and m
    n = 0
    m = 0
    size = 2*d
    while size % 2 == 0:
        n += 1
        size /= 2
    size = 2*d
    while size % 3 == 0:
        m += 1
        size /= 3

    # find the function for the unitary matrix
    # get n sets of 3 parameters for the pauli matrices
    # get m sets of 8 parameters for the gell mann matrices
    # create a dictionary
    def U(n, m, param_vals):
        '''returns the unitary matrix for given parameter values'''
        param_dict = {}
        for i in range(n):
            for j in range(3):
                param_dict[f'alpha_{i+1}_{j+1}'] = param_vals[3*i + j]
        for i in range(m):
            for j in range(8):
                param_dict[f'beta_{i+1}_{j+1}'] = param_vals[3*n + 8*i + j]

        # create the unitary matrix
        if n > 0:
            for i in range(n):
                if i == 0:
                    U = su_2(param_dict[f'alpha_{i+1}_1'], param_dict[f'alpha_{i+1}_2'], param_dict[f'alpha_{i+1}_3'])
                else:
                    U = np.kron(U, su_2(param_dict[f'alpha_{i+1}_1'], param_dict[f'alpha_{i+1}_2'], param_dict[f'alpha_{i+1}_3']))
        if m > 0:
            for i in range(m):
                if i == 0 and n == 0:
                    U = su_3(param_dict[f'beta_{i+1}_1'], param_dict[f'beta_{i+1}_2'], param_dict[f'beta_{i+1}_3'], param_dict[f'beta_{i+1}_4'], param_dict[f'beta_{i+1}_5'], param_dict[f'beta_{i+1}_6'], param_dict[f'beta_{i+1}_7'], param_dict[f'beta_{i+1}_8'])
                else:
                    U = np.kron(U, su_3(param_dict[f'beta_{i+1}_1'], param_dict[f'beta_{i+1}_2'], param_dict[f'beta_{i+1}_3'], param_dict[f'beta_{i+1}_4'], param_dict[f'beta_{i+1}_5'], param_dict[f'beta_{i+1}_6'], param_dict[f'beta_{i+1}_7'], param_dict[f'beta_{i+1}_8']))

        return U
        
    return U, n, m

def find_params(d):
    '''finds the optimal params to generate the unitary matrix'''
    # get the function to get the unitary matrix
    U, n, m = get_U(d)
    num_params = 3*n + 8*m

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
            bell_states[:,j] = make_bell(d, c, p, b=True, options='only_first', joint=False).reshape(4*d**2)
            j+=1
        # print(bell_states)
        # apply U  = \mathcal{U}^{-1} \otimes \mathcal{U}^{-1} to the bell states
        U_inv = U(n, m, param_vals).conj().T
        U_tot = np.kron(U_inv, U_inv)
        bell_proj = U_tot @ bell_states
        return num_overlapping(bell_proj)

    def prove_k(k):
        '''returns True if we can find k bell states that work, and False otherwise'''
        # get all combinations of k bell states via CNS
        # for each combination, apply U^{-1} \otimes U^{-1} to the bell states and check if the detection signatures overlap
        # if they do, return False
        # if we get through all combinations without returning False, return True

        for i in trange(comb(d**2, k)):
           # solve for the param_vals that make N = 0
            x0 = [0 for i in range(num_params)]
            # define a callback function to print the progress of the optimization

            # basinhopping
            result = basinhopping(func=N, x0=x0, minimizer_kwargs={"args": (k, i)}, niter=100, disp=False)
            # call shgo to find the minimum
            # result = shgo(func=N, args = (k, i), bounds = [(-np.pi, np.pi) for i in range(num_params)])

            print(U(n, m, result.x))
            print(U(n, m, result.x).shape)
            print(result.x)
            print(result.fun)
            if np.isclose(result.fun, 1e-10):
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
    find_params(2)

   


    
