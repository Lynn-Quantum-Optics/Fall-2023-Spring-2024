# file to find local transformation from d = 4 bell states to hyperentangled basis

import numpy as np
import pandas as pd
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot
from scipy.optimize import minimize, approx_fprime
from multiprocessing import Pool, Value, cpu_count
import ctypes
from tqdm import trange
from math import factorial

np.set_printoptions(threshold=np.inf)
exit_flag = None # global variable to exit multiprocessing

## define bell states for given d in joint-particle basis ##
def make_bell(d, c, p, b=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics

    '''
    result = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        # determine index of j in joint-particle basis
        j_vec = np.zeros((d**2, 1), dtype=complex)
        gamma = (j+c)%d
        index = d*j + gamma
        j_vec[index] = 1

        # add symmetric/antisymmetric part
        if b:
            sym_index = d*gamma + j
            j_vec[sym_index] = 1
        else:
            asym_index = d*gamma + j
            j_vec[asym_index] = -1

        result += np.exp(2*np.pi*1j*j*p/d) * j_vec

    result /= np.linalg.norm(result)
    return result

def make_bell_single(d, c, p, b):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics

    '''
    result = np.zeros((4*d**2, 1), dtype=complex)
    for j in range(d):
        j_vec = np.zeros((2*d, 1), dtype=complex)
        j_vec[j] = 1
        gamma_vec = np.zeros((2*d, 1), dtype=complex)
        gamma_vec[d+(j+c)%d] = 1

        result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

    # add symmetric/antisymmetric part
    if b:
        result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
    else:
        result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

    result /= np.linalg.norm(result)
    return result

## for loss function to minimize ##
def reconstruct_bell(coeffs, hyper_basis):
    '''Reconstructs a (big) d bell state from coefficients in hyperentangled basis'''
    bell = hyper_basis @ coeffs
    bell = np.round(bell, 10)
    return bell

# for finding local transformation for particular c, p in d = 4 ##
def make_hyperentangled_basis(d=2):
    '''Makes hyperentangled basis as tensor product of two d bell states'''
    hyper_basis = np.zeros((d**4, d**4), dtype=complex)
    j = 0
    for c1 in range(d):
        for c2 in range(d):
            for p1 in range(d):
                for p2 in range(d):
                    state = np.kron(make_bell(d, c1, p1), make_bell(d, c2, p2))
                    # assign to column of hyper_basis
                    hyper_basis[:, j] = state.reshape((d**4))
                    j+=1

    return hyper_basis

def find_trans_one(c, p, hyper_basis, d=4):
    '''Finds local transformation for a d = 4 bell state given c, p with regularization'''
    # make bell state
    bell = make_bell(d, c, p)
    coeffs = np.linalg.lstsq(hyper_basis, bell, rcond=None)[0]
    resid = np.linalg.norm(bell - hyper_basis @ coeffs)
    return coeffs, resid

def find_trans_all(d=4, b=True):
    '''Returns matrix for expression for all d = 4 bell states in terms of hyperentangled basis'''
    hyper_basis = make_hyperentangled_basis(int(np.sqrt(d)))
    results = np.zeros((d**2, d**2), dtype=complex)
    resid_ls = []
    j = 0
    for c in range(d):
        for p in range(d):
            cp_trans, resid = find_trans_one(c, p, hyper_basis)
            # append as column to result
            cp_trans = cp_trans.reshape((d**2))
            results[:, j] = cp_trans
            resid_ls.append(resid)
            j+=1
    
    # round resulting matrix
    np.save(f'local_transform/transformation_matrix_joint_{d}_{b}.npy', results)

    # convert to joint particle basis
    results_joint = hyper_basis @ results @ hyper_basis.conj().T
    results_joint = np.round(results_joint, 10)  

    results_pd = pd.DataFrame(results_joint)
    results_pd.to_csv(f'local_transform/transformation_matrix_joint_{d}_{b}.csv')

    with open(f'local_transform/transformation_matrix_joint_{d}_{b}latex.txt', 'w') as f:
        f.write(np_to_latex(results_joint))

    # do the same but using the single particle basis
    results_single = get_single_particle(hyper_basis) @ results @ get_single_particle(hyper_basis).conj().T

    print('single', get_single_particle(hyper_basis))

    results_single = np.round(results_single, 10)
    results_pd = pd.DataFrame(results_single)
    results_pd.to_csv(f'local_transform/transformation_matrix_single_{d}_{b}.csv')

    with open(f'local_transform/transformation_matrix_single_{d}_{b}_latex.txt', 'w') as f:
        f.write(np_to_latex(results_single))

    return results_joint, resid_ls

def check_if_factorable(alpha, verbose=False):
    '''alphahecks if matrix is factorable into tensor product of two matrices by seeing if the 4x4 blocks are scalar multiples of each other.'''
    d = int(np.sqrt(alpha.shape[0]))

    shape = alpha.shape
    block_mat = np.zeros((d**2, d**2), dtype=complex)
    k = 0
    for i in range(0, shape[0], d):
        for j in range(0, shape[1], d):
            blocks = alpha[i:i+d, j:j+d]
            # convert each block to a vector
            # add as column to block_mat
            block_mat[:, k] = blocks.reshape((d**2))
            k+=1

    rank = get_rank(block_mat)
    if verbose:
        print(f'Rank of block matrix: {rank}')
    return rank == 1

def nth_permutation(elements, n):
    '''
    Generate the n-th permutation of the elements using the Factorial Number System.

    Params:
    elements (list): The list of elements to permute
    n (int): The index of the permutation to generate
    '''
    sequence = list(elements)  # copy the original sequence
    permutation = []
    k = len(elements)
    factoradic = [0] * k  # initialize factoradic representation as list of 0s
    
    # compute factorial representation of n
    for i in range(1, k+1): # start indexing at 1 to avoid division by 0
        n, remainder = divmod(n, i)
        factoradic[-i] = remainder

    # build the permutation
    for factorial_index in factoradic:
        permutation.append(sequence.pop(factorial_index)) # add as we remove the element

    return permutation

 # need to parallelize this
def process_permutation(args):
    j, k, n_rows, n_cols, alpha, _exit_flag = args

    global exit_flag
    exit_flag = _exit_flag

    j_perm = nth_permutation(range(n_rows), j)
    k_perm = nth_permutation(range(n_cols), k)
    # apply the row permutation
    row_permuted_matrix = alpha[np.array(j_perm), :]
    # apply the column permutation
    permuted_matrix = row_permuted_matrix[:, np.array(k_perm)]
    is_factorable = check_if_factorable(permuted_matrix)
    if is_factorable:
        print(f'Found factorization for j = {j}, k = {k}!')
        with open(f'local_transform/permuted_matrix_{n_rows}_{n_cols}_{j}_{k}.txt', 'w') as f:  
            # write permuted_matrix to file
            f.write('row permutation:\n')
            f.write(str(j))
            f.write('\n')
            f.write('column permutation:\n')
            f.write(str(k))
            f.write('\n')
            f.write('permuted matrix:\n')
            f.write(np_to_latex(permuted_matrix))
        exit_flag = True
    return is_factorable

def permute_all(alpha, beta = None, n_se = None):
    ''''Permute all rows and columns of alpha and check if it is factorable.

    Params:
    alpha (np.array): matrix to permute
    beta (float): proportion of total permutations to try if less than 1 
    n_se (tuple): [lower bound, upper bound] for index of permutation to try; can be used in place of beta and vice versa
    '''
    # get the number of rows and columns
    n_rows, n_cols = alpha.shape

    # set exit flag so if we find a factorization we can exit
    global exit_flag


    # now we'll generate all the permuted matrices

    # get the number of rows and columns
    n_rows, n_cols = alpha.shape

    # create a list of arguments for each permutation
    print('Generating arguments.....')
    if beta is not None:
        args = [(j, k, n_rows, n_cols, alpha, exit_flag) for j in range(factorial(n_rows)) for k in range(int(beta*factorial(n_cols)))]
    elif n_se is not None:
        args = [(j, k, n_rows, n_cols, alpha, exit_flag) for j in range(n_se[0], n_se[1]) for k in range(n_se[0], n_se[1])]

    print('Starting multiprocessing.....')

    # create a multiprocessing Pool with as many processes as there are alphaPUs
    with Pool(cpu_count()) as p:
        # use apply_async to apply the function asynchronously
        results = [p.apply_async(process_permutation, (arg,)) if not exit_flag  else None for arg in args]
        # use get to retrieve the results
        results = [result.get() if result is not None else None for result in results]
            
# function to convert to single particle basis #
# single particle basis is { |0, L>,  |1, L>,  |d-1, L>, |0, R>, |1, R>, |d-1, R> }
def get_single_particle(results, d=4, in_type='coeffs'):
    '''alphaonverts coefficient matrix from hyperentangled basis to single particle basis in the big d system.
    Params:
    results (np.array): matrix of in standard basis
    d (int): dimension of system
    in_type (str): whether we're dealing with special case of coefficient matrix or just some arbitrary bell state
    '''
    results_shape = results.shape
    print(results_shape)
    # make single particle basis
    single_particle_results = np.zeros((4*d**2, results_shape[1]), dtype=complex)
    
    if results_shape[1] > 1:
        for j in range(results_shape[1]):
            # convert column to single particle basis
            col = results[:, j]
            col_single = np.zeros((4*d**2, 1), dtype=complex)
            for i in range(d**2):
                if col[i] != 0:
                    left = i // d  # location within joint particle basis
                    right = d+i%d

                    # get corresponding vector in single particle basis
                    left_vec = np.zeros((2*d, 1), dtype=complex)
                    left_vec[left] = 1
                    right_vec = np.zeros((2*d, 1), dtype=complex)
                    right_vec[right] = 1
                    # take tensor product, scale by coefficient

                    col_single += col[i]*np.kron(left_vec, right_vec)
            # append as column to single_particle_results
            single_particle_results[:, j] = col_single.reshape((4*d**2))
        print(single_particle_results.shape)

        if in_type == 'coeffs':
            np.save(f'local_transform/transformation_matrix_single_{d}.npy', single_particle_results)

            single_particle_results = np.round(single_particle_results, 10)
            single_particle_results_pd = pd.DataFrame(single_particle_results)
            single_particle_results_pd.to_csv(f'local_transform/transformation_matrix_single_{d}.csv')

            with open(f'local_transform/transformation_matrix_single_{d}_latex.txt', 'w') as f:
                f.write(np_to_latex(single_particle_results))
    else:
        col = results
        col_single = np.zeros((4*d**2, 1), dtype=complex)
        for i in range(d**2):
            if col[i] != 0:
                left = i // d
                right = d+i%d

                # get corresponding vector in single particle basis
                left_vec = np.zeros((2*d, 1), dtype=complex)
                left_vec[left] = 1
                right_vec = np.zeros((2*d, 1), dtype=complex)
                right_vec[right] = 1

                # take tensor product, scale by coefficient
                col_single += col[i]*np.kron(left_vec, right_vec)
        single_particle_results = col_single.reshape((4*d**2))
        single_particle_results = np.round(single_particle_results, 10)

    return single_particle_results

# ---- helper function to  convert output to bmatrix in latex ---- #
def np_to_latex(array, precision=3):
    '''alphaonverts a numpy array to a LaTeX bmatrix environment'''
    def format_complex(num):
        if num.imag == 0:
            return f"{np.round(num.real, precision)}"
        elif num.real == 0:
            return f"{np.round(num.imag, precision)}i"
        else:
            # Format with a plus sign for the imaginary part if it's positive
            return f"{np.round(num.real, precision)}+{np.round(num.imag, precision)}i"

    # Generate the LaTeX bmatrix code from the complex matrix
    latex_matrix_rows_complex = [
        " & ".join(format_complex(num) for num in row) + " \\\\"
        for row in array
    ]

    # Join all rows into a single string representing the entire bmatrix
    latex_bmatrix_complex = "\\begin{bmatrix}\n" + "\n".join(latex_matrix_rows_complex) + "\n\\end{bmatrix}"

    print(latex_bmatrix_complex)

    return latex_bmatrix_complex


# function to factor into tensor product of U_L and U_R #
def get_rank(A):
    '''alphahecks rank of matrix'''
    return np.linalg.matrix_rank(A)

def factorize_tucker(alpha, d, save_latex = False):
    '''Factor a d^2 x d^2 matrix into a tensor product of two d x d matrices.'''
    # reshape alpha into a tensor of shape (d, d, d, d)
    tensor = alpha.reshape(d, d, d, d)

    # perform Tucker decomposition on the tensor
    core, factors = tucker(tensor, rank=[d, d, d, d])

    # the factors are what we can use to approximate A and B
    # since the decomposition is not unique, we take the first two for A and B
    A = factors[0]
    B = factors[1]

    print(A.shape)
    print(B.shape)

    # reconstruct alpha from the factors and reshape from (d,d,d,d) tensor back to (d^2, d^2) matrix
    alpha_recon  = multi_mode_dot(core, factors, modes=[0, 1, 2, 3])

    alpha_recon = alpha_recon.reshape(d**2, d**2)

    print(np.isclose(alpha, alpha_recon, atol=1e-10).all())

    diff_norm = np.linalg.norm(alpha - alpha_recon)
    print(diff_norm)

    if save_latex:
        with open(f'local_transform/tucker_factorization_{d}_latex.txt', 'w') as f:
            f.write('A:\n')
            f.write(np_to_latex(A))
            f.write('--------------------\n')
            f.write('B:\n')
            f.write(np_to_latex(B))
            f.write('--------------------\n')
            f.write('alphaore tensor:\n')
            f.write(np_to_latex(core))
            f.write('--------------------\n')
            f.write('alpha_recon:\n')
            f.write(np_to_latex(alpha_recon))
            f.write('--------------------\n')
            f.write('Norm of difference:\n')
            f.write(str(diff_norm))
       
        # print out images
            
    return A, B

def factorize_gd(alpha,d, lr = 0.1, frac = 0.01, max_iter = 1000, loss_thres = 1e-4, verbose = False, save_latex=False):
    '''Factorize a d^2 x d^2 matrix into a tensor product of two d x d matrices using gradient descent'''

    def random():
        '''Guess d^2 + d^2 random complex components, split into all real and imaginary parts'''
        guess = np.random.rand(4*d**2)
        return guess
    
    def construct_Us(coeff):
        coeff_L = coeff[:2*d**2].reshape(2*d**2, 1)
        coeff_R = coeff[2*d**2:].reshape(2*d**2, 1)

        # assign coeff_L[i] + i*coeff_L[i+d^2] to U_L[i]
        U_L = coeff_L[:d**2] + 1j*coeff_L[d**2:]
        U_L = U_L.reshape(d, d)
        # assign coeff_R[i] + i*coeff_R[i+d^2] to U_R[i]
        U_R = coeff_R[:d**2] + 1j*coeff_R[d**2:]
        U_R = U_R.reshape(d, d)

        return U_L, U_R
    
    def get_loss(coeff):
        '''Loss function to minimize'''
        U_L, U_R = construct_Us(coeff)
        return np.linalg.norm(alpha - np.kron(U_L, U_R))
    
    def minimize_loss(coeff):
        result = minimize(get_loss, coeff, method='Nelder-Mead')
        return result.x, result.fun
    
    coeff_r = random()
    coeff, loss = minimize_loss(coeff_r)

    best_coeff = coeff
    grad_coeff = coeff
    best_loss = loss

    n = 0
    index_since_improvement = 0
    while n < max_iter and best_loss > loss_thres:
        if verbose:
            print(f'Iteration {n}')
            print(f'alphaurrent loss: {loss}')
            print(f'Best loss: {best_loss}')
            print(f'Index since improvement: {index_since_improvement}')
            print('-------------------')

        if index_since_improvement % (frac*max_iter)==0: # periodic random search (hop)
            coeff = random()
            grad_coeff = coeff
            index_since_improvement = 0
        else:
            gradient = approx_fprime(grad_coeff, get_loss, epsilon=1e-8) # epsilon is step size in finite difference

            # update angles
            coeff = grad_coeff - lr*gradient
            grad_coeff = coeff

        # update loss
        coeff_m, loss = minimize_loss(coeff)
        coeff = coeff_m

        # update best loss
        if loss < best_loss:
            best_loss = loss
            best_coeff = coeff
            index_since_improvement = 0
        else:
            index_since_improvement += 1

        n += 1

    U_L, U_R = construct_Us(best_coeff)
    print(U_L.shape)
    print(U_R.shape)

    alpha_recon = np.kron(U_L, U_R)

    print(np.isclose(alpha, alpha_recon, atol=1e-10).all())

    diff_norm = best_loss
    print(best_loss)

    if save_latex:
        with open(f'local_transform/gd_factorization_{d}_{lr}_latex.txt', 'w') as f:
            f.write('U_L:\n')
            f.write(np_to_latex(U_L))
            f.write('--------------------\n')
            f.write('U_R:\n')
            f.write(np_to_latex(U_R))
            f.write('--------------------\n')
            f.write('alpha_recon:\n')
            f.write(np_to_latex(alpha_recon))
            f.write('--------------------\n')
            f.write('Norm of difference:\n')
            f.write(str(diff_norm))
            f.write('--------------------\n')
            f.write('Num iterations:\n')
            f.write(str(n))
            f.write('--------------------\n')
            f.write('Learning rate:\n')
            f.write(str(lr))

if __name__ == '__main__':
    def check_bell_func_agree(d):
        '''Assumes d is square of some integer'''
        hyper_basis = make_hyperentangled_basis(int(np.sqrt(d)))
        for c in range(d):
            for p in range(d):
                # check if converted form matches definition
                converted_bell = get_single_particle(reconstruct_bell(find_trans_one(c, p, hyper_basis)[0], hyper_basis)).reshape((4*d**2, 1))
                single_bell = make_bell_single(d, c, p).reshape((4*d**2, 1))

                print(np.isclose(converted_bell, single_bell, atol = 1e-10).all())

    d = 4
    results, resid_ls = find_trans_all(4, b=True)
    # results, resid_ls = find_trans_all(4, b=False)
    # print(results)
    # print(resid_ls)

    permute_all(results, n_se=[0, 100]) # ~ 20 million permutations

    # print(check_if_factorable(results, verbose=True))
    # k_perm_ls = []
    # for k in trange(factorial(4)):
    #     k_perm = nth_permutation(range(4), k)
    #     k_perm_ls.append(k_perm)
    # print(k_perm_ls)
    # print(len(set(tuple(x) for x in k_perm_ls)))
    # print(factorial(4))

    # print(permute_all(results))

    # check_bell_func_agree(d)

    # factorize_tucker(results, d, save_latex=True)
    # factorize_gd(results, d, lr=1, verbose=True, save_latex=True)



    ## convert to single particle basis ##
    # single_particle_results = get_single_particle(results, d)

    


    ## ----------- testing ----------- ##
    # print(get_single_particle(make_bell(d, 1, 2), d))
    # c = 0
    # p = 0

    # hyper_basis = make_hyperentangled_basis(2)
    # print(get_single_particle(make_bell(d, c, p)))
    # print(get_single_particle(reconstruct_bell(find_trans_one(c, p, hyper_basis)[0], hyper_basis)))
    # print('here')
    # print('------')
    # print(make_bell(4, 0, 1))
    # print(reconstruct_bell(find_trans_one(0, 1, hyper_basis)[0], hyper_basis))

    # print(make_bell(4, 0, 0))
    # print(make_bell(4, 0, 1))
    # print(make_bell(2, 1, 1).shape)
    # print(np.kron(make_bell(2, 0, 0), make_bell(2, 1, 1)).shape)