# file to find local transformation from d = 4 bell states to hyperentangled basis

import numpy as np
import pandas as pd
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot
from scipy.optimize import minimize, approx_fprime
from multiprocessing import Pool, cpu_count
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(threshold=np.inf)
exit_flag = None # global variable to exit multiprocessing

## define bell states for given d in joint-particle basis ##
def make_bell(d, c, p, b=True, options = 'none', joint=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    options (str): whether to include symmetric/antisymmetric part term or both
    single (bool): whether to calculate in joint basis or joint

    '''
    if joint:
        result = np.zeros((d**2, 1), dtype=complex)
        for j in range(d):
            # determine index of j in joint-particle basis
            j_vec = np.zeros((d**2, 1), dtype=complex)
            gamma = (j+c)%d
            first_index = d*j + gamma
            second_index = d*gamma + j

            # basis is in form |j, gamma>, where j is fixed when gamma is varied
            if options=='none':
                # proceed normally
                j_vec[first_index] = np.exp(2*np.pi*1j*j*p/d)

                # add symmetric/antisymmetric part: swap the L and R particles, so now gamma is fixed and j is varied
                if b: 
                    j_vec[second_index] = np.exp(2*np.pi*1j*j*p/d)
                else:
                    j_vec[second_index] = -np.exp(2*np.pi*1j*j*p/d)

            elif options == 'only_first': # only the first term
                j_vec[first_index] = np.exp(2*np.pi*1j*j*p/d)

            elif options == 'only_second': # only the second term
                if b:
                    j_vec[second_index] = np.exp(2*np.pi*1j*j*p/d)
                else:
                    j_vec[second_index] = -np.exp(2*np.pi*1j*j*p/d)
            else:
                raise ValueError(f'Invalid option flag: {options}. Valid options are: none, only_first, only_second')

            # add to result
            result += j_vec
    else:
        result = np.zeros((4*d**2, 1), dtype=complex)
        for j in range(d):
            j_vec = np.zeros((2*d, 1), dtype=complex)
            j_vec[j] = 1
            gamma_vec = np.zeros((2*d, 1), dtype=complex)
            gamma_vec[d+(j+c)%d] = 1

            if options == 'none':
                result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)
                # add symmetric/antisymmetric part
                if b:
                    result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
                else:
                    result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

            elif options == 'only_first': # only the first erm
                result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

            elif options == 'only_second': # only the second term
                # add symmetric/antisymmetric part
                if b:
                    result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
                else:
                    result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

    # normalize
    result /= np.linalg.norm(result)
    return result

def get_bell_matrix(d, b=True, options='none', joint=True):
    '''Builds complete matrix of bell states for given d and options.

    Parameters:
        d (int): dimension of system
        c (int): correlation class
        p (int): phase class
        b (bool): whether to assume bosonic or fermionic statistics
        options (str): whether to include symmetric/antisymmetric part term or both
        single (bool): whether to calculate in joint basis or joint
        
        
    
    '''
    for c in range(d):
        for p in range(d):
            bell = make_bell(d=d, c=c, p=p, b=b, options=options, joint=joint)
            if c == 0 and p == 0:
                result = bell
            else:
                result = np.hstack((result, bell))
    return result

def eigen_decomp(matrix):
    '''Finds the number of non-overlapping eigenvectors.'''
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # round eigenvectors to 10 decimal places
    eigenvectors = np.round(eigenvectors, 10)
    # round eigenvalues to 10 decimal places
    eigenvalues = np.abs(np.round(eigenvalues, 10))
    
    
    # convert eigenvectors to matrix, where each column is an eigenvector
    eigenvectors = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=complex)    
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i].reshape((matrix.shape[0]))

    gauss = gaus_elim(eigenvectors)
    print(num_nonoverlapping(gauss))

def num_nonoverlapping(vectors):
    # converts each vector to a binary array of whether it is zero or not
    # then take dot product of each vector with all other vectors and sum
    # divide by 2 to account for double counting
    # subtract from total number of vectors

    # convert to binary
    vectors = np.where(np.isclose(vectors, 0, rtol=1e-10), 0, 1)
    print(vectors)
    # go through each vector and dot product with all other vectors
    num_overlapping = 0
    for i in range(vectors.shape[1]):
        for j in range(i+1, vectors.shape[1]):
            print(np.dot(vectors[:, i], vectors[:, j]))
            num_overlapping += np.dot(vectors[:, i], vectors[:, j])
    print('num overlap', num_overlapping)
    return vectors.shape[1] - num_overlapping

def gaus_elim(m):
    h, w = m.shape

    for i in range(min(h, w)):
        maxrow = np.argmax(np.abs(m[i:, i])) + i

        m[[i, maxrow]] = m[[maxrow, i]]

        if m[i, i] == 0:
            continue

        m[i] = m[i] / m[i, i]

        for j in range(i + 1, h):
            m[j] = m[j] - m[j, i] * m[i]

    m = np.round(m, 10)

    # get column vectors
    cols = [m[:, i] for i in range(m.shape[1])]
    print(num_nonoverlapping(cols))


    # m = np.array(m, dtype=complex)
    # print(np.linalg.matrix_rank(m))

    return m

# function to convert to single particle basis #
# single particle basis is { |0, L>,  |1, L>,  |d-1, L>, |0, R>, |1, R>, |d-1, R> }
def get_single_particle(results, d=4, in_type='coeffs'):
    '''converts coefficient matrix from hyperentangled basis to single particle basis in the big d system.
    Params:
    results (np.array): matrix of in standard basis
    d (int): dimension of system
    in_type (str): whether we're dealing with special case of coefficient matrix or just some arbitrary bell state
    '''
    results_shape = results.shape
    # print(results_shape)
    # make single particle basis
    single_particle_results = np.zeros((4*d**2, results_shape[1]), dtype=complex)
    
    if results_shape[1] > 1:
        for j in range(results_shape[1]):
            # convert column to single particle basis
            col = results[:, j]
            col_single = np.zeros((4*d**2, 1), dtype=complex)
            # check number of nonzero elements
            num_nonzero = np.count_nonzero(col)
            if num_nonzero == d: # need to account for c = 0 case
                do_extra_sym = True
            else:
                do_extra_sym = False
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

                    single_i = np.kron(left_vec, right_vec)
                    if do_extra_sym:
                        single_i += np.kron(right_vec, left_vec) 

                    col_single += col[i]/np.sqrt(2)*single_i
            # append as column to single_particle_results
            single_particle_results[:, j] = col_single.reshape((4*d**2))
        # print(single_particle_results.shape)

        if in_type == 'coeffs':
            np.save(f'local_transform/transformation_matrix_single_{d}.npy', single_particle_results)

            single_particle_results = np.round(single_particle_results, 10)
            single_particle_results_pd = pd.DataFrame(single_particle_results)
            single_particle_results_pd.to_csv(f'local_transform/transformation_matrix_single_{d}.csv')

            with open(f'local_transform/transformation_matrix_single_{d}_latex.txt', 'w') as f:
                f.write(np_to_latex(single_particle_results))
    else:
        col = results
        print('initial', results.T)
        # check number of nonzero elements
        num_nonzero = np.count_nonzero(col)
        if num_nonzero == d: # need to account for c = 0 case
            do_extra_sym = True
        else:
            do_extra_sym = False
        col_single = np.zeros((4*d**2, 1), dtype=complex)
        for i in range(col.shape[0]):
            if col[i] != 0:
                left = i // d  # location within joint particle basis
                right = d+(i%d)
                
                # get corresponding vector in single particle basis
                left_vec = np.zeros((2*d, 1), dtype=complex)
                left_vec[left] = 1
                right_vec = np.zeros((2*d, 1), dtype=complex)
                right_vec[right] = 1

                single_i = np.kron(left_vec, right_vec)
                if do_extra_sym:
                    single_i += np.kron(right_vec, left_vec) 

                print(i, single_i.T)

                col_single += col[i]*single_i

                
        single_particle_results = col_single.reshape((4*d**2))
        # normalize
        single_particle_results /= np.linalg.norm(single_particle_results)

    return single_particle_results

## for loss function to minimize ##
def reconstruct_bell(coeffs, hyper_basis):
    '''Reconstructs a (big) d bell state from coefficients in hyperentangled basis'''
    bell = hyper_basis @ coeffs
    bell = np.round(bell)
    return bell

# for finding local transformation for particular c, p in d = 4 ##
def make_hyperentangled_basis(d=2, joint=True):
    '''Makes hyperentangled basis as tensor product of two d bell states.

    Parameters:
        d (int): dimension of system
        b (bool): whether to assume bosonic or fermionic statistics
        options (str): whether to include symmetric/antisymmetric part term or both
        joint (bool): whether to calculate in joint basis or single

    NOTE: assumes not anti/symmetric part
    '''
    if joint:
        hyper_basis = np.zeros((d**4, d**4), dtype=complex)
    else:
        hyper_basis = np.zeros((16*d**4, d**4), dtype=complex)
    j = 0
    for c1 in range(d):
        for c2 in range(d):
            for p1 in range(d):
                for p2 in range(d):
                    state = np.kron(make_bell(d=d, c=c1, p=p1, options='only_first', joint=joint), make_bell(d=d, c=c2, p=p2,   options='only_first', joint=joint))
                    # assign to column of hyper_basis
                    if joint:
                        hyper_basis[:, j] = state.reshape((d**4))
                    else:
                        hyper_basis[:, j] = state.reshape((16*d**4))

                    j+=1

    return hyper_basis

def find_trans_one(c, p, b=True, hyper_basis=None, d=4, options='none'):
    '''Finds local transformation for a d = 4 bell state given c, p with regularization'''
    # make bell state
    bell = make_bell(d=d, c=c, p=p, b=b, options=options)
    coeffs = np.linalg.lstsq(hyper_basis, bell, rcond=None)[0]
    resid = np.linalg.norm(bell - hyper_basis @ coeffs)
    return coeffs, resid

def find_trans_all(d=4, b=True, options='none'):
    '''Returns matrix for expression for all d = 4 bell states in terms of hyperentangled basis'''
    hyper_basis = make_hyperentangled_basis(d=int(np.sqrt(d)))
    print_matrix(hyper_basis, title=f'Hyperentangled Basis in Joint, {d}')
    np.save(f'local_transform/hyper_basis_{d}_{b}_{options}.npy', hyper_basis)

    results = np.zeros((d**2, d**2), dtype=complex)
    resid_ls = []
    j = 0
    for c in range(d):
        for p in range(d):
            cp_trans, resid = find_trans_one(c=c, p=p, b=b, hyper_basis=hyper_basis,d= d, options = options)
            # append as column to result
            cp_trans = cp_trans.reshape((d**2))
            results[:, j] = cp_trans
            resid_ls.append(resid)
            j+=1
    
    # round resulting matrix
    np.save(f'local_transform/transformation_matrix_hyper_{d}_{b}_{options}.npy', results)
    print_matrix(results, title=f'Hyperentangled Alpha, {b}, {options}')
    np.save(f'local_transform/results_hyper_{d}_{b}_{options}.npy', results)

    is_factorable = check_if_factorable(results, verbose=True)
    print(f'Is hyperfactorable: {is_factorable}')

    # convert to joint particle basis
    results_joint = hyper_basis @ results @ hyper_basis.conj().T
    results_pd = pd.DataFrame(results_joint)
    results_pd.to_csv(f'local_transform/transformation_matrix_joint_{d}_{b}_{options}.csv')
    print_matrix(results_joint, title=f'Joint Particle Alpha, {b}, {options}, {d}')
    np.save(f'local_transform/results_joint_{d}_{b}_{options}.npy', results_joint)

    is_factorable = check_if_factorable(results_joint, verbose=True)
    print(f'Is jointfactorable: {is_factorable}')

    with open(f'local_transform/transformation_matrix_joint_{d}_{b}_{options}latex.txt', 'w') as f:
        f.write(np_to_latex(results_joint))

    # do the same but using the single particle basis
    # print('joint', results_joint)

    # FIX THIS BY CONVERTING EACH COLUMN TO SINGLE PARTICLE BASIS SINCE WE HAVE SOMETHING IN THE JOINT ALREADY.
    single_hyper_basis = make_hyperentangled_basis(d=int(np.sqrt(d)), joint=False)
    print_matrix(single_hyper_basis, title=f'Hyperentangled Basis in Single, {d}')
    np.save(f'local_transform/single_hyper_basis_{d}_{b}_{options}.npy', single_hyper_basis)

    results_single = single_hyper_basis @ results @ single_hyper_basis.conj().T
    results_pd = pd.DataFrame(results_single)
    results_pd.to_csv(f'local_transform/transformation_matrix_single_{d}_{b}_{options}.csv')
    print_matrix(results_single, title=f'Single Particle Alpha, {b}, {options}, {d}')
    np.save(f'local_transform/results_single_{d}_{b}_{options}.npy', results_single)

    is_factorable = check_if_factorable(results_single, verbose=True)
    print(f'Is singlefactorable: {is_factorable}')

    with open(f'local_transform/transformation_matrix_single_{d}_{b}_{options}_latex.txt', 'w') as f:
        f.write(np_to_latex(results_single))

    return results_joint, results_single, resid_ls

def check_if_factorable(alpha, verbose=False):
    '''alpha checks if matrix is factorable into tensor product of two matrices by seeing if the 4x4 blocks are scalar multiples of each other.'''
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

## ------ automatic checking of permutations ------ ##
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

## ------ manual checking of permutations ------ ##            
def find_all_nonzero(alpha):
    '''Returns all nonzero elements of alpha and number of repeats'''

    print(np.linalg.det(alpha))
    print(np.trace(alpha))

    non_zero = []
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if alpha[i, j] != 0:
                non_zero.append(alpha[i, j])
    
    # get number of repeats
    unique = list(set(non_zero))
    # find the frequency of each unique
    freq = [non_zero.count(x) for x in unique]
    
    # couple into a tuple
    tot_non_zero = list(zip(unique, freq))
    return tot_non_zero, np.sort_complex(non_zero)

def is_permutation(alpha, beta):
    '''Checks if beta is a permutation of alpha.'''
    if alpha.shape != beta.shape:
        return False
    if not(np.isclose(np.linalg.det(alpha), np.linalg.det(beta), atol=1e-10)):
        print(f'Determinants not equal. You have: {np.linalg.det(beta)} for permuted')
        return False
    if not(np.isclose(np.trace(alpha), np.trace(beta), atol=1e-10)):
        print(f'Traces not equal. You have: {np.trace(beta)} for permuted')
        return False
    
    # check if all nonzero elements are the same
    _,alpha_nonzero  = find_all_nonzero(alpha)
    _, beta_nonzero = find_all_nonzero(beta)

    if alpha_nonzero != beta_nonzero:
        print('Nonzero elements not equal')
        return False
    
    return True

def try_guess_factorization(blocks, indices, alpha):
    '''Construct matrix based on blocks and see if it is a permutation of rows and cols of alpha.
    
    Params:
    blocks (list): list of unique 4x4 blocks
    indices (list): list index of block type in blocks
    '''

    d = int(np.sqrt(alpha.shape[0]))

    # construct matrix
    permuted = np.zeros((d**2, d**2), dtype=complex)
    for i, index in enumerate(indices):
        row = (i // d)* d
        col = (i % d)* d
        
        permuted[row:row+d, col:col+d] = blocks[index]
        print(blocks[index])

    # check if permuted is a permutation of alpha
    return is_permutation(alpha, permuted)

# ---- helper function to  convert output to bmatrix in latex and to print ---- #
def np_to_latex(array, precision=3):
    '''alphaonverts a numpy array to a LaTeX bmatrix environment'''
    def format_complex(num):
        if num.imag == 0:
            return f"{np.round(num.real, precision)}"
        elif num.real == 0:
            return f"{np.round(num.imag, precision)}i"
        elif num.imag != 0 and num.real != 0:
            # Format with a plus sign for the imaginary part if it's positive
            return f"{np.round(num.real, precision)}+{np.round(num.imag, precision)}i"
        else:
            return "0"

    # Generate the LaTeX bmatrix code from the complex matrix
    latex_matrix_rows_complex = [
        " & ".join(format_complex(num) for num in row) + " \\\\"
        for row in array
    ]

    # Join all rows into a single string representing the entire bmatrix
    latex_bmatrix_complex = "\\begin{bmatrix}\n" + "\n".join(latex_matrix_rows_complex) + "\n\\end{bmatrix}"

    # print(latex_bmatrix_complex)

    return latex_bmatrix_complex

def print_matrix(array, title=None, show=False, label=True):
    '''Converts matrix to matplotlib image with mag and phase separate.

    Params:
    array (np.array): matrix to convert
    title (str): title of plot
    show (bool): whether to show plot
    label (bool): whether to include labels

    Returns:
    None, but saves plot to local_transform folder
    
    '''
    mag = np.abs(array)
    phase = np.angle(array)

    # wherever mag is 0, set phase to 0
    phase[mag == 0] = 0

    # include colorbar
    if mag.shape[1] > 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, ax = plt.subplots(1, 2)
    cax0 = ax[0].matshow(mag)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cax0, cax = cax)
    ax[0].set_title('Magnitude')
    if not label:
        ax[0].set_xticks([])
        ax[0].set_yticks([])

    cax1 = ax[1].matshow(phase)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cax1, cax=cax)
    ax[1].set_title('Phase')
    if not label:
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'local_transform/{title}.pdf')
    if show:
        plt.show()

# function to factor into tensor product of U_L and U_R #
def get_rank(A):
    '''checks rank of matrix'''
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

# check if computed single particle hyperentangled basis matches known in single #
def check_hyperentangled_basis(d=4, b=True, options='none'):
    '''Checks if hyperentangled basis matches known basis in single tensor single particle basis.

    Params:
    d (int): dimension of states in hyperentangled system
    b (bool): whether to assume bosonic or fermionic statistics
    options (str): whether to include symmetric/antisymmetric part term or both
    
    '''
    # iterate through all possible c, p
    for c in range(d):
        for p in range(d):
            bell_joint = make_bell(d=d, c=c, p=p, b=b, options=options, joint=True)
            converted = get_single_particle(bell_joint, d=d, in_type='bell')
            converted = converted.reshape((4*d**2, 1))
            bell_single = make_bell(d=d, c=c, p=p, b=b, options=options, joint=False)
            bell_single = bell_single.reshape((4*d**2, 1))
            converted = np.round(converted, 10)
            bell_single = np.round(bell_single, 10)
            diff = np.linalg.norm(converted - bell_single)
            print(diff)
            if diff > 1e-10:
                print(f'c = {c}, p = {p}')
                print('Converted:')
                print(converted.T)
                print('Actual:')
                print(bell_single.T)
                print('-----------------')
                return False
    return True

if __name__ == '__main__':
    d = 4

    bell_mat = get_bell_matrix(d=d, b=True, options='only_first', joint=True)
    eigen_decomp(bell_mat)
    # print(gaus_elim(bell_mat))



    # results_joint, results_single, resid_ls = find_trans_all(d=4, b=False, options='none')
    # check_hyperentangled_basis(d, b=True)
    

    # hyper_basis = make_hyperentangled_basis(int(np.sqrt(d)))
    # print_matrix(hyper_basis, title='Hyperentangled Basis')

    # for c in range(d):
    #     for p in range(d):
    #         print(f'c = {c}, p = {p}')
    #         print(make_bell(d=d, c=c, p=p, b=False, options='none').T)
    #         print('-------')

    # print(find_all_nonzero(results)[1])

    ## test factorization for d= 4##
    # plus = 0.5+0.5*1j
    # minus = 0.5-0.5*1j
    # base = np.array([[1, 0, 0, 0], [0, 2,0,0], [0,0,2,0], [0,0,0,1]])
    # # base = np.eye(4)
    # # factors = [0, 1, 0.5+0.5*1j, 0.5+0.5*1j]
    # print(np.linalg.det(minus*base))
    # print(np.trace(minus*base))
    
    # indices = [
    #             1, 0, 0, 0,
    #             0, 0, 0, 0,
    #             0, 0, 1, 0,
    #             0, 0, 0, 0
    #           ]

    # try_guess_factorization(base, factors, indices, results)