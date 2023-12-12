# file to try out Lynn's idea of symmetrized bell states

import numpy as np
from u_parametrize import *

def check_entangled(bell, display_val=False):
    '''Computes reduced density matrix for each particle and checks if they are mixed.

    Note: performs the partial trace over the second subsystem.

    :bell: d^2 x 1 vector
    :display_val: boolean to display the reduced density matrix and the trace of the rdm squared
    
    '''
    # get d from the bell state and make sure it's normalized
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2))
    bell /= np.linalg.norm(bell)

    # get density matrix
    rho = np.outer(bell, np.conj(bell))

    # get reduced density matrix
    rho_reduced = np.zeros((d, d), dtype=complex)

    # partial trace over the second subsystem
    for i in range(d):
        # projector for each basis state of the second subsystem
        basis_state = np.zeros(d)
        basis_state[i] = 1
        projector = np.kron(basis_state, np.eye(d))

        # add to reduced density matrix
        rho_reduced += np.dot(np.dot(projector, rho), projector.T)

    # normalize!
    rho_reduced = np.trace(rho_reduced) * rho_reduced
    
    # expect 1/d for fully entangled system
    tr_sq =  np.trace(rho_reduced @ rho_reduced)
    if display_val:
        # print(rho_reduced)
        print(tr_sq)
    return np.isclose(tr_sq, 1/d, 1e-10)

def bell_us(d, c, p):
    '''Function to generate a bell state in the joint particle basis in the old way.

    Params:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    
    '''
    bell = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        L = j
        R = (j+c) % d
        index = L*d + R
        bell[index] = np.exp(2*np.pi*1j*p*j / d)

    # normalize
    bell /= np.sqrt(d)

    return bell

def bell_s_2(d, c, p):
    '''Function to generate a bell state in the joint particle basis that specifically is symmetric.

    Params:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    
    '''
    bell = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        # only mess with terms for c!= 0 or d/2
        if c == 0 or (d % 2 == 0 and j == d//2):
            L = j
            R = (j+c) % d
            index = L*d + R
            bell[index] = np.exp(2*np.pi*1j*p*j / d)
        else:
            # use the c = 1, p = 0 bell state as basis to tweak
            effective_c = 1
            # effective_p = 0
            L = j
            R = (j+effective_c) % d
            index = L*d + R
            index_swap = R*d + L
            if c < d//2: # keep symmetrized; minus sign goes where j is 
                # p tells us where to put minus sign
                if j == p: # put minus here
                    bell[index] = -1
                    bell[index_swap] = -1
                else:
                    bell[index] = 1
                    bell[index_swap] = 1
            else: # anti-symmetrize
                # print(p, j)
                if j == p: # put minus here
                    bell[index] = -1
                    bell[index_swap] = 1
                else:
                    bell[index] = 1
                    bell[index_swap] = -1
    # print('-----')
    # print(f'c = {c}, p = {p}, bell = {display_bell(np.round(bell, 10))}')     
    # normalize
    bell /= np.linalg.norm(bell)
    return bell

def bell_s(d, c, p):
    '''Function to generate a bell state in the joint particle basis that specifically is symmetric.

    Params:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    
    '''
    bell = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        L = j
        R = (j+c) % d
        index = L*d + R
        bell[index] = np.exp(2*np.pi*1j*p*j / d)
        if c!= 0 or (d % 2 == 0 and j == d//2):
            index = R*d + L
            bell[index] = np.exp(2*np.pi*1j*p*j / d)
            if p %2 == 0: # if even p
                bell[0] = np.exp(2*np.pi*1j*p*j / d)

    # normalize
    bell /= np.linalg.norm(bell)
    return bell

def symmetric(bell=None, display=False):
    '''Checks if an input bell state (in joint particle basis) is symmetric wrt particle exchange.

    Params:
        :bell: d^2 x 1 vector
        :display: boolean to display initial and transformed

    Note: by construction, the L state corresponds to the index // d, R is index % d. Thus, index = L*d + R
    '''
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2, 1))
    bell /= np.linalg.norm(bell)

    # need to swap L and R and check if they are the same
    new_bell = np.zeros((d**2,1), dtype=complex)
    for i, b in enumerate(bell):
        if b != 0:
            L = i // d
            R = i % d
            new_i = int(R*d + L)
            new_bell[new_i] = b
    
    # account for overall phase
    # find phase of first nonzero term
    first_non0 = bell[np.nonzero(bell)[0][0]]
    imag_orig = np.imag(first_non0)
    phase_orig = np.angle(first_non0)
    phase_new = np.angle(new_bell[np.nonzero(new_bell)[0][0]])

    if phase_orig != phase_new and np.isclose(imag_orig, 0, 1e-10):
        new_bell *= np.exp(1j*(phase_orig - phase_new))

    if display:
        print(bell)
        print('------')
        print(new_bell)

    return np.all(np.isclose(bell, new_bell, 1e-10))

def find_cp(d, bell_func = bell_us, operation=symmetric):
    '''Finds what c and p for given d will yield symmetric state'''
    valid_cp = []
    for c in range(d):
        for p in range(d):
            if operation(bell_func(d, c, p)):
                valid_cp.append((c, p))

    print(f'Num valid c, p: {len(valid_cp)}')
    return valid_cp
        
def check_all_bell(d, func = None, bell_gen_func = bell_us):
    '''Performs the function func on each of the bell states for a given dimension d.

    Params:
        :d: dimension of the system
        :func: function to perform on each bell state. default is None, which just prints the bell states without doing anything
        :bell_gen_func: function to generate the bell states. default is bell_us, which generates the bell states in the old way
    '''

    for c in range(d):
        for p in range(d):
            print(c, p)
            if func is not None:
                print(func(bell_gen_func(d, c, p)))
            else:
                print(bell_gen_func(d, c, p), display_val=True)
            print('-----')

def check_all_entangled(d, bell_gen_func = bell_s):
    '''Performs the function func on each of the bell states for a given dimension d.

    Params:
        :d: dimension of the system
        :func: function to perform on each bell state. default is None, which just prints the bell states without doing anything
        :bell_gen_func: function to generate the bell states. default is bell_us, which generates the bell states in the old way
    '''
    entangled = 0

    for c in range(d):
        for p in range(d):
            print(c, p)
            ent = check_entangled(bell_gen_func(d, c, p), display_val=True)
            if ent == True:
                entangled += 1
            print('-----')

    return entangled

def display_bell(bell):
    '''Converts bell state as vector and prints it in bra ket form.'''
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2, 1))
    bell /= np.linalg.norm(bell)
    bell_str = ''
    for i, b in enumerate(bell):
        if b != 0:
            L = i // d
            R = i % d
            if bell_str == '':
                bell_str+=f'{np.round(b[0],3)} |{L}>|{R}>'
            else:
                bell_str+=f'+ {np.round(b[0],3)} |{L}>|{R}>'
    return bell_str

if __name__ == '__main__':
    d = 4
    print(check_entangled(bell_s(d, 3, 2), display_val=True))
    # print(display_bell(bell_s(d, 3, 2)))



    # print('Symmetrizing')
    # print(find_cp(d, bell_s, check_entangled))
    # print(find_cp(d, bell_s, symmetric))
    # print('-----')
    # print('Us')
    # print(find_cp(d, bell_us, check_entangled))
    # print(find_cp(d, bell_us, symmetric))
    # check_all_bell(d, display_bell, bell_s)
    # print(check_all_entangled(d))
    # find_params(d, bell_s)