# file to test action of U and determine resulting detection signatures 

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
def givens_rotation(n, i, j, theta, phi):
    """
    Creates an n-dimensional Givens rotation matrix for dimensions i and j,
    with rotation angle theta and phase phi.
    """
    if i > j: i, j = j, i
    assert 0 <= i < j < n

    G = np.identity(n, dtype=complex)
    G[i, i] = np.cos(theta)
    G[j, j] = np.cos(theta)
    G[i, j] = -np.exp(1j * phi) * np.sin(theta)
    G[j, i] = np.exp(-1j * phi) * np.sin(theta)
    return G

def diagonal_phase(n, phases):
    """
    Creates an n-dimensional diagonal phase matrix given a list of phases.
    """
    print(phases)
    assert len(phases) == n-1
    
    # insert phase of 0 at beginning; this is to account for the fact that global phase is irrelevant
    phases = np.array(phases)
    phases = phases.reshape((n-1, 1))
    phases = np.insert(phases, 0, 0)
    return np.diag(np.exp(1j * phases))

def su_n_matrix(n, params):
    """
    Constructs an SU(n) matrix given parameters for Givens rotations and phases.
    givens_params should be a list of tuples (i, j, theta, phi)
    phase_params should be a list of phases of length n-1
    """
    # split params into givens and phases
    givens_params, phase_params = params[:-n+1], params[-n+1:]
    U = np.identity(n, dtype=complex)
    for i, j, theta, phi in givens_params:
        G = givens_rotation(n, i, j, theta, phi)
        U = np.dot(U, G)
    P = diagonal_phase(n, phase_params)
    U = np.dot(U, P)
    return U

def non_entangling_unitary(n, params):
    '''
    Creates a non-entangling unitary operation for a system of dimension d.
    '''
    U1 = su_n_matrix(n, params)
    U2 = su_n_matrix(n, params)
    return np.kron(U1, U2)

def random_params(n):
    """
    Generates a list of random angles of length 2 (theta, phi) for given rotation + n -1 for diagonal= n+1.
    """
     # Randomly choose Givens rotation parameters
    givens_params = [(i, j, 
                      np.random.uniform(0, 2 * np.pi), 
                      np.random.uniform(0, 2 * np.pi))
                     for i in range(n) for j in range(i+1, n)]

    # Randomly choose phase parameters
    phase_params = np.random.uniform(0, 2 * np.pi, n-1)
    return givens_params + list(phase_params)

# ------ define detection operation -------- #
def detection_sig(bell, U):
    '''Returns detection signature of bell state after action of U.

    Parameters:
        :bell: bell state
        :U: unitary operator
    '''
    sig = U @ bell
    return sig

def check_overlap(signatures):
    '''
    Checks if there is any overlap in the elements of the detection signatures.
    '''
    c = 0 # counter
    flattened_sigs = [set(np.flatnonzero(sig)) for sig in signatures]
    for i, sig_i in enumerate(flattened_sigs):
        for j, sig_j in enumerate(flattened_sigs):
            if i != j and sig_i.intersection(sig_j):
                c+=1  # Overlap found
    return c

# def get_k(d, k, c_ls, p_ls):
#     ''''
#     Computes the total overlap for a given group of k states, specified by the correlation and phase classes.
#     '''


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

if __name__ == '__main__':
    d = 6
    bell = get_bell(d, 0, 0)
    params = random_params(d)
    U = non_entangling_unitary(d, params)
    detec = detection_sig(bell, U)
    print(detec.T)


