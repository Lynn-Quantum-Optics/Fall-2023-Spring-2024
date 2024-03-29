# file to test action of U and determine resulting detection signatures 

import numpy as np
from math import factorial
from scipy import stats
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from oscars_toolbox.trabbit import trabbit

# ---------- define bell states ------------- #
def get_bell(d, c, p):
    '''Returns hyperdimensional bell state using original definition.

    Parameters:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    '''

    state = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        j_vec = np.zeros((d**2, 1), dtype=complex)
        gamma = (j+c)%d
        index = d*j + gamma

        j_vec[index] = np.exp(1j*2*np.pi*p*j/d)

        state += j_vec

    state /= np.linalg.norm(state)

    return state

# ------ generate random unitary ----------- #
def generate_random_unitary_no_param(d):
    '''
    Generates a random d x d unitary matrix.
    '''
    return stats.unitary_group.rvs(d)

def non_entangling_unitary_no_param(d):
    '''
    Creates a non-entangling unitary operation for a system of dimension d.
    '''
    U1 = generate_random_unitary_no_param(d)
    U2 = generate_random_unitary_no_param(d)
    return np.kron(U1, U2)

# ------ generate random unitary with parameters ----------- #
def givens_rotation(d, i, j, theta, phi):
    """
    Creates an n-dimensional Givens rotation matrix for dimensions i and j,
    with rotation angle theta and phase phi.
    """
    if i > j: i, j = j, i
    assert 0 <= i < j < d

    G = np.identity(d, dtype=complex)
    G[i, i] = np.cos(theta)
    G[j, j] = np.cos(theta)
    G[i, j] = -np.exp(1j * phi) * np.sin(theta)
    G[j, i] = np.exp(-1j * phi) * np.sin(theta)
    return G

def diagonal_phase(d, phases):
    """
    Creates an d-dimensional diagonal phase matrix given a list of phases.
    """
    assert len(phases) == d-1
    
    # insert phase of 0 at beginning; this is to account for the fact that global phase is irrelevant
    phases = np.array(phases)
    phases = phases.reshape((d-1, 1))
    phases = np.insert(phases, 0, 0)
    return np.diag(np.exp(1j * phases))

def pair_to_index(a, b, d):
    '''Helper matrix to get the index of a pair of elements in array [(a,b) for a < b <= d]'''
    # Validate that a < b < n
    if not (0 <= a < b < d):
        raise ValueError("Invalid pair (a, b). Must satisfy 0 <= a < b < n.")
    
    # Calculate the index
    return int((a * (2 * d - a - 1)) // 2 + (b - a - 1))

def su_n_matrix(d, params):
    """
    Constructs an SU(d) matrix given parameters for Givens rotations and phases.
    givens_params should be a list of tuples (i, j, theta, phi)
    phase_params should be a list of phases of length d-1
    """
    # split params into givens and phases
    split_index = d*(d-1)//2
    givens_params, phase_params = params[:-d+1], params[-d+1:]
    U = np.identity(d, dtype=complex)
    for i in range(d):
        for j in range(i+1, d):
            theta = givens_params[pair_to_index(i, j, d)]
            phi = givens_params[pair_to_index(i, j, d) + split_index]
            U = givens_rotation(d, i, j, theta, phi) @ U
    
    P = diagonal_phase(d, phase_params)
    U = P @ U
    return np.kron(U, U)

def non_entangling_unitary(n, params):
    '''
    Creates a non-entangling unitary operation for a system of dimension d.
    '''
    U1 = su_n_matrix(n, params)
    U2 = su_n_matrix(n, params)
    return np.kron(U1, U2)

def random_params(d):
    """
    Generates a list of random angles of length 2*d^2 (theta, phi) for given rotation + n -1 for diagonal= n+1.
    """
     # Randomly choose Givens rotation parameters
    givens_params = np.zeros(d*(d-1))
    for i in range(d):
        for j in range(i+1, d):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            givens_params[pair_to_index(i, j, d)] = theta
            givens_params[pair_to_index(i, j, d) + d*(d-1)//2] = phi

    # Randomly choose phase parameters
    phase_params = np.random.uniform(0, 2 * np.pi, d-1)
    return list(givens_params) + list(phase_params)

# ------ define detection operation -------- #
def detection_sig(bell, U):
    '''Returns detection signature of bell state after action of U.

    Parameters:
        :bell: bell state
        :U: unitary operator
    '''
    sig = U @ bell
    return sig

def check_overlap(signatures, ret_c=True):
    '''
    Checks if there is any overlap in the elements of the detection signatures.

    Parameters:
        :signatures: list of detection signatures
        :ret_c: whether to return the number of overlaps, or to sum the vals at the overlaps
    '''
    assert len(signatures) >= 2, f'Need at least two signatures to check overlap. Got {len(signatures)}.'
    if ret_c:
        c = 0 # counter
    else:
        val = 0 # sum of vals
    for i in range(len(signatures)):
        for j in range(i+1, len(signatures)): # compare all pairs
            sig0 = signatures[i]
            sig1 = signatures[j]
            for k in range(len(sig0)):
                if not(np.isclose(sig0[k], 0, rtol=1e-10)) and not(np.isclose(sig1[k], 0, rtol=1e-10)):
                    if ret_c:
                        c += 1
                    else:
                        val += np.abs(sig0[k])**2
                        val += np.abs(sig1[k])**2
    if ret_c:
        return c
    else:
        return val

def get_k(d, U, k_states, ret_c = True, display=False):
    ''''
    Computes the total overlap for a given group of k states, specified by the correlation and phase classes.

    Parameters:
        :d: dimension of the system
        :U: unitary operator
        :k_states: list of the bell states
        :ret_c: whether to return the number of overlaps, or to sum the vals at the overlaps
        :display: whether to display the detection signatures as a matrix
    '''
    # get detection signatures as matrix
    sigs = np.array([detection_sig(bell, U) for bell in k_states])
    if display:
        # display matrix
        display_matrix(sigs.reshape(len(sigs), d**2).T)
        print(f'Overlap: {check_overlap(sigs, ret_c=ret_c)}')
    else:
        # check overlap

        overlap =  check_overlap(sigs, ret_c=ret_c)
        return overlap

# ------ optimize U based on guess -------- #
def U_guess(params, d):
    '''Returns matrix based on Lynn's guess that each column has entries like v_i = sqrt(q/d^2)e^{i 2 pi r / (2d)} where q and r are integers to be determined.'''
    assert len(params) == 2*d**2, f'Need 2d^2 parameters. Got {len(params)}.'

    # convert to ints
    # params = [int(param) for param in params]

    # first d^2 params are q
    q_ls = params[:d**2]
    # second d^2 params are r
    r_ls = params[d**2:]
    # zip together
    qr_ls = list(zip(q_ls, r_ls))

    # print(qr_ls)

    # create matrix
    try:
        U = [np.sqrt(q/d)*np.exp(1j*2*np.pi*r/(d)) for q, r in qr_ls]
        U = np.array(U).reshape((d, d))
        U = np.kron(U, U)
    except RuntimeError:
        raise ValueError(f'Invalid parameters: {params}')

    return U

from sympy import Matrix, sqrt, exp, I, pi, zeros, symbols, pprint

def U_guess_sympy(d):
    '''Returns matrix based on Lynn's guess using sympy for symbolic computation.'''

    # Create symbolic parameters for q and r
    q_ls = symbols(f'q0:{d**2}')
    r_ls = symbols(f'r0:{d**2}')

    # Zip together
    qr_ls = zip(q_ls, r_ls)

    # Create matrix using sympy
    U = zeros(d, d)
    for i, (q, r) in enumerate(qr_ls):
        row, col = divmod(i, d)
        U[row, col] = sqrt(q/d) * exp(I * 2 * pi * r / d)

    # Use kronecker product (tensor product) for the final matrix
    U = Matrix(np.kron(U, U))

    return U

def rand_seq_to_sum(n, use_int=True):
    '''Generates random sequence of numbers of size n that sum to n.

    Parameters:
        :n: number of sum to and size of sequence
        :use_int: whether to use integers or float. if integers, uses sequential alg
     
      '''
    if use_int:
        # initialize list
        seq = [0]*n
        # generate random sequence of numbers of size n that sum to n
        for j in range(n):
            # choose random index
            i = np.random.randint(0, n+1) # make sure to include n
            while i + sum(seq) > n:
                i = np.random.randint(0, n+1) # make sure to include n
            seq[j] += i
    else: # use floats. need sum to be n
         # Generate initial random floats
        seq = np.random.random(n) * n
        current_sum = sum(seq)

        precision = 10

        # Iteratively adjust to sum to n
        while round(current_sum, precision) != n:
            for i in range(n):
                if current_sum < n:
                    increment = min(np.random.random() * (n - current_sum), n - current_sum)
                    seq[i] += increment
                    current_sum += increment
                elif current_sum > n:
                    decrement = min(seq[i], current_sum - n)
                    seq[i] -= decrement
                    current_sum -= decrement

                # Adjust precision to match desired level
                current_sum = round(current_sum, precision)
                
                if current_sum == n:
                    break
        # check all positive
        assert all([s >= 0 for s in seq]), f'Sequence has negative numbers: {seq}'

        # shuffle order
        np.random.shuffle(seq)
       
    return seq

def random_guess(d, use_int=True):
    '''Gets random integers for q and r.

    Parameters:
        :d: dimension of the system
        :use_int: whether to use integers for q and r or float
    '''
    guess = []
    # q ranges from 0 to d^2 such that sum of q is d^2
    # do this need d times
    for _ in range(d):
        guess += list(rand_seq_to_sum(d, use_int=use_int))
    # r ranges from 0 to d such that sum of r is d
    # do this need d times
    for _ in range(d):
        guess += list(rand_seq_to_sum(d, use_int=use_int))
    return guess

def loss(params, d, k_states, ret_c):
    '''Loss function for optimization based on a parametrized get_k function that logs the c and p ls.'''
    U = U_guess(params, d)
    overlap_loss = get_k(d, U, k_states=k_states, ret_c=ret_c)
    # # need to ensure sum of each d length sequence is d
    # sum_loss = 0
    # for i in range(0, len(params), d):
    #     sum_loss += np.abs(sum(params[i:i+d]) - d)
    # need to ensure unitary
    unitary_loss = np.linalg.norm(np.eye(d**2) - U @ U.conj().T)
    return overlap_loss + unitary_loss

def optimization_task(loss_func, random_func, bounds, initial_guess=None):
    return trabbit(loss_func, random_func, bounds=bounds, alpha=0.8, temperature=0.01, x0_ls=initial_guess)

def find_params(d, combinations, use_int=True, parallel=False, initial_guess=None):
    '''Uses trabbit to find parameters for U.

    Parameters:
        :d: dimension of the system
        :combinations: list of tuples of correlation and phase classes
        :use_int: whether to use integers for q and r or float
        :parallel: whether to run in parallel
    '''
    # get the states
    k_states = [get_bell(d, c, p) for c, p in combinations]
    random_func = partial(random_guess, d=d, use_int=use_int)
    loss_func = partial(loss, d=d, k_states = k_states, ret_c=False)
    bounds = [(0, d)]*2*d**2
    opt_task = partial(optimization_task, loss_func=loss_func, random_func=random_func, bounds=bounds, initial_guess=initial_guess)


    if not parallel:
        x_best, best_loss = opt_task()
        print(f'Best params: {list(x_best)}')
        print(f'Best loss: {best_loss}')
        return x_best
    
    else:
        print(f'Running in parallel with {ProcessPoolExecutor()._max_workers} workers.')
        # Create a pool of workers using all available CPUs
        with ProcessPoolExecutor() as executor:  # Defaults to the number of CPUs
            futures = [executor.submit(opt_task) for _ in range(executor._max_workers)]

            results = []
            for future in futures:
                x_best, best_loss = future.result()
                results.append((x_best, best_loss))

        # Process results
        best_result = min(results, key=lambda x: x[1])
        print(f'Best params: {list(best_result[0])}')
        print(f'Best loss: {best_result[1]}')
        return best_result[0]

# ------ optimize based on given rotations -------- #
def loss_gr(params, d, combinations, ret_c):
    ''' Loss function for optimization based on a parametrized get_k function for c and p ls.'''
    U = su_n_matrix(d, params)
    return get_k(d, U, combinations=combinations, ret_c=ret_c)

def find_params_gr(d, combinations):
    '''Uses trabbit to find parameters for U.'''
    random_func = partial(random_params, d)
    loss_func = partial(loss_gr, d=d, combinations=combinations, ret_c=True)
    # bounds = [(0, 2*np.pi)]*d*(d-1) + [(0, 2*np.pi)]*(d-1)
    x_best, best_loss = trabbit(loss_func, random_func, alpha=0.8, temperature=0.01)
    print(f'Best params: {list(x_best)}')
    print(f'Best loss: {best_loss}')
    return x_best

# ------ brute force find params -------- #
def count_partitions(number, max_number):
    if number == 0:
        return [[]]
    partitions = []
    for i in range(min(number, max_number), 0, -1):
        for p in count_partitions(number - i, i):
            partitions.append([i] + p)
    return partitions

def permutations_with_zeros(partition, vector_length):
    # Number of zeros to fill the remaining positions
    num_zeros = vector_length - len(partition)

    # Updated partition including zeros
    updated_partition = partition + [0] * num_zeros

    # Calculate permutations with repetitions
    total_elements = len(updated_partition)
    repetitions = {number: updated_partition.count(number) for number in set(updated_partition)}
    permutation_count = factorial(total_elements)
    for rep in repetitions.values():
        permutation_count //= factorial(rep)
    
    return permutation_count

def get_total_permutations(d, partitions):
    # Calculate total permutations with zeros included
    partitions = count_partitions(d, d)
    return sum(permutations_with_zeros(partition, d) for partition in partitions)

# ------- for display -------- #
def display_bell(bell, print_bell=True):
    '''Converts bell state as vector and prints it in bra ket form.'''
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2, 1))
    bell /= np.linalg.norm(bell)
    bell_str = ''
    for i, b in enumerate(bell):
        if b != 0:
            L = i // d
            R = i % d
            # convert coeff to mag, angle
            mag = np.abs(b[0]).real
            phase = np.angle(b[0]).real
            if bell_str == '':
                bell_str+=f'{mag}e^{phase}*1j |{L}>|{R}>'
            else:
                bell_str+=f'+ {mag}e^{phase}*1j |{L}>|{R}>'
    if print_bell:
        print(bell_str)
    return bell_str

def display_matrix(mat, save=False, filename=None):
    '''Displays matrix as a heatmap.'''
    mag = np.abs(mat)
    phase = np.angle(mat)
    # where mag is 0, phase is 0
    mag[np.isclose(mag, 0, 1e-10)] = 0
    phase[np.isclose(phase, 0, 1e-10)] = 0
    phase[mag==0] = 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # plot mag and phase with colorbar and labels
    im1 = ax[0].imshow(mag)
    ax[0].set_title('Magnitude')
    fig.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(phase)
    ax[1].set_title('Phase')
    fig.colorbar(im2, ax=ax[1])

    if save and filename is not None:
        plt.savefig(filename)

    plt.show()

def validate_params(params, d, combinations, comb='test'):
    '''Validates that params determined through optimization are valid.

    Parameters:
        :params: list of parameters for U
        :d: dimension of the system
        :combinations: list of tuples of correlation and phase classes
        :comb: int specifying which combination to test    
    '''
    # first get U
    U = U_guess(params, d)

    # print out U
    display_matrix(U)

    np.save(f'saved_U/U_{d}_{comb}.npy', U)

    # check that U is unitary
    assert np.allclose(U @ U.conj().T, np.eye(d**2), rtol=1e-8), f'U is not unitary. U @ U.conj().T = {U @ U.conj().T}'

    # check that U is non-entangling
    # assert np.linalg.det(U) == 1, f'U is entangling. det(U) = {np.linalg.det(U)}'

    # get detection sig
    get_k(d, U, combinations=combinations, display=True)


if __name__ == '__main__':
    from time import time
    print('starting test...')
    d = 6
    ## define combinations ##
    comb0 = [(c, 0) for c in range(d)]
    comb1 = [(c, 0) for c in range(d)] + [(0, 1)]

    t0 = time()
    print(f'start time is {t0}')
    find_params(d, combinations=comb1, use_int=False, parallel=False, initial_guess=[[1.235161204888607, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5619366850069816, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.79724804715832, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.7972472701687545, 0.0, 0.6709094054450491, 0.0, 0.6709094942628906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.797210930528499, 0.0, 2.211717043124469, 3.2286898023563553, 0.0, 0.0, 0.0, 1.5275063215320808, 2.5706380357671747e-06, 1.324407500510177, 1.9047651160721664, 2.376593075020871, 0.5815963661854644, 0.2310332140388418, 1.94693364806412, 0.0, 0.08044476052774212, 0.0, 5.052915084053701, 0.0, 0.20959970128123784, 0.5005873049358195, 1.0798317880176238, 0.9092232201052284, 4.148162619805921, 1.0538725811735773e-06, 0.0, 0.0, 2.850900179354211, 0.999826202195426, 0.05219291665468766, 3.2789323217963044, 1.009528460815879, 0.778521837283892, 0.045591264096812724, 1.5614087675653368, 0.519420716354121, 2.0855276254346578]])
    tf = time()
    print(f'end time is {tf}')
    print(f'time elapsed is {tf-t0}')

    ## ------- for testing found results ------- ##
    ''' key:
        comb0: (c, 0) for c in range(d)
        comb1: (c, 0) for c in range(d) + (0,1)     
    '''

    # found 1/8/23 on sysphus in aws
    comb0_params = [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06100355776038493, 0.0, 0.0, 3.7248492191224716, 2.458762840804276, 0.0, 0.0, 0.2690569544938626, 4.053622109167405, 4.4408920985006275e-08, 0.0, 2.593739880961608, 2.360520705449435, 0.21669792765696938, 0.0, 0.0, 0.03459782962561464, 4.601070763856098, 0.09842334271826958, 1.6490709678608918e-23, 0.0, 4.974659685099812, 0.40160657935127436, 0.7067750947370453, 1.1578330235381022, 0.0011762260070042117, 0.0, 4.842166976523448, 0.0, 2.316850708921771, 0.0, 3.1115361301974387, 0.0, 2.888463869745582, 0.0, 0.24109763075311855]

    # validate_params(comb0_params, d, combinations=[(c, 0) for c in range(d)])





