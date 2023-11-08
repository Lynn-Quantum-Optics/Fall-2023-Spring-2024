# file to find local transformation from d = 4 bell states to hyperentangled basis

import numpy as np
import pandas as pd
from tensorly.decomposition import tucker
from scipy.optimize import minimize, approx_fprime

## define bell states for given d in joint-particle basis ##
def make_bell(d, c, p):
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

        # # add symmetric/antisymmetric part
        # if b:
        #     sym_index = d*gamma + j
        #     j_vec[sym_index] = 1
        # else:
        #     asym_index = d*gamma + j
        #     j_vec[asym_index] = -1

        result += np.exp(2*np.pi*1j*j*p/d) * j_vec

    result /= np.sqrt(d)
    return result

def make_bell_single(d, c, p):
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

    result /= np.sqrt(d)
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

def find_trans_one(c, p, hyper_basis, d=4, alpha=0.0001):
    '''Finds local transformation for a d = 4 bell state given c, p with regularization'''
    # make bell state
    bell = make_bell(d, c, p)
    coeffs = np.linalg.lstsq(hyper_basis, bell, rcond=None)[0]
    resid = np.linalg.norm(bell - hyper_basis @ coeffs)
    return coeffs, resid

def find_trans_all(d=4):
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
    np.save(f'local_transform/transformation_matrix_joint_{d}.npy', results)

    results = np.round(results, 10)
    results_pd = pd.DataFrame(results)
    results_pd.to_csv(f'local_transform/transformation_matrix_joint_{d}.csv')

    with open(f'local_transform/transformation_matrix_joint_{d}_latex.txt', 'w') as f:
        f.write(np_to_latex(results))

    return results, resid_ls

# function to convert to single particle basis #
# single particle basis is { |0, L>,  |1, L>,  |d-1, L>, |0, R>, |1, R>, |d-1, R> }
def get_single_particle(results, d=4, in_type='coeffs'):
    '''Converts coefficient matrix from hyperentangled basis to single particle basis in the big d system.
    Params:
    results (np.array): matrix of in standard basis
    d (int): dimension of system
    in_type (str): whether we're dealing with special case of coefficient matrix or just some arbitrary bell state
    '''
    results_shape = results.shape
    # make single particle basis
    single_particle_results = np.zeros((4*d**2, results_shape[1]), dtype=complex)
    
    if results_shape[1] > 1:
        for j in range(d**2):
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
def np_to_latex(array, precision=2):
    '''Converts a numpy array to a LaTeX bmatrix environment'''
    def format_complex(c, precision):
        '''Formats a complex number as a string'''
        if not(np.isclose(c.real, 0)) and not(np.isclose(c.imag, 0)):
            return format(c.real, f".{precision}f") + " + " + format(c.imag, f".{precision}f") + "i"
        elif np.isclose(c.imag, 0) and not(np.isclose(c.real, 0)):
            return format(c.real, f".{precision}f")
        elif np.isclose(c.real, 0) and not(np.isclose(c.imag, 0)):
            return format(c.imag, f".{precision}f") + "i"
        else:
            return "0"
    
    latex_str = "\\begin{bmatrix}\n"
    for row in array:
        row_str = " & ".join(format_complex(x, precision) for x in row)
        latex_str += row_str + " \\\\\n"
    latex_str += "\\end{bmatrix}"
    return latex_str

# function to factor into tensor product of U_L and U_R #
def get_rank(A):
    '''Checks rank of matrix'''
    return np.linalg.matrix_rank(A)

def factorize_tucker(C, d, save_latex = False):
    '''Factor a d^2 x d^2 matrix into a tensor product of two d x d matrices.'''
    # Reshape C into a tensor of shape (d, d, d, d)
    tensor = C.reshape(d, d, d, d)
    print(d)

    # Perform Tucker decomposition on the tensor
    core, factors = tucker(tensor, rank=[d, d, d, d])

    # The factors are what we can use to approximate A and B
    # Since the decomposition is not unique, we take the first two for A and B
    A = factors[0]
    B = factors[1]

    print(A.shape)
    print(B.shape)

    C_recon = np.kron(A, B)

    print(np.isclose(C, C_recon, atol=1e-10).all())

    diff_norm = np.linalg.norm(C - C_recon)
    print(diff_norm)

    if save_latex:
        with open(f'local_transform/tucker_factorization_{d}_latex.txt', 'w') as f:
            f.write('A:\n')
            f.write(np_to_latex(A))
            f.write('B:\n')
            f.write(np_to_latex(B))
            f.write('C_recon:\n')
            f.write(np_to_latex(C_recon))
            f.write('Norm of difference:\n')
            f.write(str(diff_norm))


    return A, B

def factorize_svd(C, d, save_latex = False):
    '''Factor a d^2 x d^2 matrix into a tensor product of two d x d matrices.'''
    # Reshape the matrix C into a 4D tensor with shape (d, d, d, d)
    tensor = C.reshape(d, d, d, d)
    
    # Unfold the tensor into matrices
    matrices = [tensor.transpose(i, j, k, l).reshape(d**2, d**2) for i, j, k, l in [(0, 2, 1, 3), (0, 3, 1, 2)]]
    
    # Apply SVD to each matrix and take the component with the largest singular value
    components = [np.linalg.svd(mat, full_matrices=False)[0][:, 0].reshape(d, d) for mat in matrices]
    
    # Normalize the components
    norm_factor = np.sqrt(np.linalg.norm(components[0]) * np.linalg.norm(components[1]))
    A = components[0] / norm_factor
    B = components[1] / norm_factor

    print(A.shape)
    print(B.shape)

    C_recon = np.kron(A, B)

    print(np.isclose(C, C_recon, atol=1e-10).all())

    diff_norm = np.linalg.norm(C - C_recon)
    print(diff_norm)

    if save_latex:
        with open(f'local_transform/svd_factorization_{d}_latex.txt', 'w') as f:
            f.write('A:\n')
            f.write(np_to_latex(A))
            f.write('B:\n')
            f.write(np_to_latex(B))
            f.write('C_recon:\n')
            f.write(np_to_latex(C_recon))
            f.write('Norm of difference:\n')
            f.write(str(diff_norm))
    
    return A, B

def factorize_gd(C,d, alpha = 0.1, frac = 0.01, max_iter = 1000, loss_thres = 1e-4, verbose = False, save_latex=False):
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
        return np.linalg.norm(C - np.kron(U_L, U_R))
    
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
            print(f'Current loss: {loss}')
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
            coeff = grad_coeff - alpha*gradient
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

    C_recon = np.kron(U_L, U_R)

    print(np.isclose(C, C_recon, atol=1e-10).all())

    diff_norm = best_loss
    print(best_loss)

    if save_latex:
        with open(f'local_transform/gd_factorization_{d}_{alpha}_latex.txt', 'w') as f:
            f.write('U_L:\n')
            f.write(np_to_latex(U_L))
            f.write('U_R:\n')
            f.write(np_to_latex(U_R))
            f.write('C_recon:\n')
            f.write(np_to_latex(C_recon))
            f.write('Norm of difference:\n')
            f.write(str(diff_norm))
            f.write('Num iterations:\n')
            f.write(str(n))
            f.write('')
            f.write('Learning rate:\n')
            f.write(str(alpha))

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
    # check_bell_func_agree(d)

    results, resid_ls = find_trans_all(4)
    print(results)
    print(resid_ls)

    # factorize_tucker(results, d, save_latex=True)
    # factorize_svd(results, d, save_latex=True)
    factorize_gd(results, d, alpha=.1, verbose=True, save_latex=True)
    # print(hosvd(results))



    # # convert to single particle basis
    # single_particle_results = get_single_particle(results, d)

    



           
            

    # factor into tensor product of U_L and U_R
    # print(factorize_tensor(results, d))
    


    

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