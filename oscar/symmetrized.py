# file to try out Lynn's idea of symmetrized bell states

import numpy as np
from functools import partial
from u_parametrize import *
from trabbit import trabbit
import matplotlib.pyplot as plt

def print_matrix(matrix, title=None):
    '''Prints matrix using imshow'''
    mag = np.abs(matrix)
    phase = np.angle(matrix)

    # where mag is 0, phase is 0
    phase = np.where(np.isclose(mag, 0, 1e-10), 0, phase)

    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(mag)
    ax[0].set_title('Magnitude')
    ax[0].axis('off')
    ax[1].imshow(phase)
    ax[1].set_title('Phase')
    ax[1].axis('off')

    if title is not None:
        plt.suptitle(title)
        plt.savefig('figs/' + title + '.pdf')
    else:
        plt.show()

def check_entangled(bell, ret_val = False, display_val=False):
    '''Computes reduced density matrix for each particle and checks if they are mixed.

    Note: performs the partial trace over the second subsystem.

    :bell: d^2 x 1 vector
    :ret_val: boolean to return the trace of rdm squared
    :display_val: boolean to display the reduced density matrix and the trace of the rdm squared
    
    '''
    # get d from the bell state and make sure it's normalized
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2))
    bell /= np.linalg.norm(bell)

    # get density matrix
    rho = bell @ np.conj(bell)

    # get reduced density matrix
    rho_reduced = np.zeros((d, d), dtype=complex)

    Id = np.eye(d)

    # partial trace over the second subsystem
    for i in range(d):
        # projector for each basis state of the second subsystem
        basis_state = np.zeros(d)
        basis_state[i] = 1
        projector = np.kron(basis_state, Id)

        # add to reduced density matrix
        rho_reduced += projector @ rho @ projector.T

    # normalize!
    rho_reduced = np.trace(rho_reduced) * rho_reduced
    
    # expect 1/d for fully entangled system
    tr_sq =  np.trace(rho_reduced @ rho_reduced)
    if display_val:
        # print(rho_reduced)
        print(tr_sq)
    if ret_val:
        return tr_sq
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

def bell_s(d, c, p):
    '''Function to generate a bell state in the joint particle basis that specifically is symmetric.

    Params:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    
    '''
    bell = np.zeros((d**2, 1), dtype=complex)
    exp_initial = 2*np.pi*1j*p / d
    for j in range(d):
        # bell_j = np.zeros((d**2, 1), dtype=complex)
        L = j
        R = (j+c) % d
        index_1 = L*d + R
        # phase = np.exp(2*np.pi*1j*j*p / d)
        
        # print(np.abs(phase), np.angle(phase))
        # phase = np.exp(np.pi*1j*j*(p -3) / d)
        # phase = np.exp(2 * np.pi * 1j * (j * c**2 + p**2) / d)
        if c % (d//2) == 0:
            phase = np.exp(exp_initial + (-1)**j*2*np.pi/(d//2))
            print((-1)**j*2*np.pi/(d//2))
        

        if c!= 0 or (d % 2 == 0 and j == d//2):
            index_2 = R*d + L
            bell[index_1] = phase
            bell[index_2] = phase
            # if p % 2 == 0: # if even p
            #     bell[0] = phase
        else:
            phase = np.exp(np.pi*1j*j*(p-1) / d)
            bell[index_1] = phase


    # normalize
    bell /= np.linalg.norm(bell)
    return bell

def bell_s2(d, c, p):
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

def bell_s3(d, c, p):
    '''Function to generate a symmetric Bell state in the joint particle basis.

    Params:
        :d: dimension of the system
        :c: correlation class
        :p: phase class
    '''
    bell = np.zeros((d**2, 1), dtype=complex)    
    for j in range(d):
        L = j
        R = (j+c) % d
        index1 = L*d + R
        index2 = R*d + L
        phase = np.exp(2*np.pi*1j*j*((p - 1) // 2) / d)

        # symmetrize
        bell[index1] = phase
        bell[index2] = phase

    # normalize 
    bell /= np.linalg.norm(bell)
    return bell

def symmetric(bell=None, ret_norm=False, display=False):
    '''Checks if an input bell state (in joint particle basis) is symmetric wrt particle exchange.

    Params:
        :bell: d^2 x 1 vector
        :ret_norm: boolean to return the norm of the bell state - new bell
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

    if ret_norm:
        return np.linalg.norm(bell - new_bell)

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

def check_all_entangled(d, bell_gen_func = bell_s, odd_p = False, prop_c = False):
    '''Performs the function func on each of the bell states for a given dimension d.

    Params:
        :d: dimension of the system
        :func: function to perform on each bell state. default is None, which just prints the bell states without doing anything
        :bell_gen_func: function to generate the bell states. default is bell_us, which generates the bell states in the old way
        :odd_p: whether to use odd only p or normal 0 -> d-1 p. default is False, which uses normal p
        :prop_c: whether to use c = 0, 1, ..., d-1 or just c = n * d//2. default is False, which uses normal c
    '''
    entangled = 0

    if odd_p:
        p_ls = [2*n+1 for n in range(d)]
        # p_ls = np.array([2*n+1 for n in range(d)])+4
        # # remove 3 and add one more odd
        # p_ls.remove(3)
        # p_ls.append(2*d+1)
    else:
        p_ls = range(d)

    if prop_c:
        c_ls = [n*d//2 for n in range(d)]
    else:
        c_ls = range(d)

    for c in c_ls:
        for p in p_ls:
            print(c, p)
            ent = check_entangled(bell_gen_func(d, c, p), display_val=True)
            if ent == True:
                entangled += 1
            print('-----')

    return entangled == d**2

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
            # convert coeff to mag, angle
            mag = np.abs(b[0]).real
            phase = np.angle(b[0]).real
            if bell_str == '':
                bell_str+=f'{mag}e^{phase}*1j |{L}>|{R}>'
            else:
                bell_str+=f'+ {mag}e^{phase}*1j |{L}>|{R}>'
    return bell_str

def convert_bell_str(bell_str, d):
    '''Converts bell state in bra ket form to vector.'''
    bell = np.zeros((d**2, 1), dtype=complex)
    bell_str = bell_str.split('+')
    for term in bell_str:
        # split by space to get coeff and ket
        term = term.split()
        coeff = term[0]
        # get rid of e^
        if 'e' in coeff:
            coeff = coeff.split('e^')
            mag = float(coeff[0])
            angle = float(coeff[1].split('*')[0])
            coeff = mag*np.exp(angle * 1j)
        ket_pair = term[1]
        # get L and R values
        ket_pair = ket_pair.split('>')
        L = int(ket_pair[0][1:])
        R = int(ket_pair[1][1:])
        
        # get index
        index = L*d + R
        bell[index] = coeff
    return bell

entanglement = partial(check_entangled, ret_val=True)
symmet = partial(symmetric, ret_norm=True)

def loss_ent(x, bell):
    bell = bell.reshape((d**2,))
    # convert to full vector
    vec = x[:d**2] + 1j*x[d**2:]
    bell += vec
    # normalize
    bell /= np.linalg.norm(bell)
    ent = entanglement(bell)
    sym = symmet(bell)
    ent_real = np.real(ent)
    ent_imag = np.imag(ent)
    targ_ent = 1/d
    return (ent_real - targ_ent)**2 + (ent_imag)**2 + sym**2

# def loss_orth(x, d):
#     # find the inner products of all bell states
#     bell_ls = []
#     for c in range(d):
#         for p in range(d):
#             bell_ls.append(bell_s(d, c, p))
            
#     # convert to full vector
#     vec = x[:d**2] + 1j*x[d**2:]
#     bell_ls = np.array(bell_ls)

def make_entangled(bell, x0_ls = None):
    '''Uses GD to find the state we need to add to the bell state to make it fully entangled'''
    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2,))
    bell /= np.linalg.norm(bell)
    
    def random_gen():
        '''Separate real and imaginary parts of vector and normalize'''
        vec = np.concatenate([np.random.rand(d**2), + np.random.rand(d**2)])
        combined_vec = vec[:d**2] + 1j*vec[d**2:]
        return vec/np.linalg.norm(combined_vec)
    
    # try to find solution
    loss_bell = partial(loss, bell=bell)
    x_best, loss_best = trabbit(loss_bell, random_gen, verbose=True, frac=0.01, alpha=.7, tol=1e-10, x0_ls=x0_ls)
    print(loss_best)
    print(list(x_best))
    return x_best

def correct_bell(bell, corr_str):
    '''Takes in a str of the form 'coeff1 |L1>|R1> + coeff2 |L2>|R2> + ...' and returns the corrected bell state.

    Params:
        :bell: d^2 x 1 vector
        :corr_str: string of the form 'coeff1 |L1>|R1> + coeff2 |L2>|R2> + ...'    
    '''

    d = int(np.sqrt(len(bell)))
    bell = bell.reshape((d**2,))
    bell /= np.linalg.norm(bell)

    corr = convert_bell_str(corr_str, d)
    corr = corr.reshape((d**2,))
    bell += corr
    bell /= np.linalg.norm(bell)
    return bell

def get_d_primes(d):
    '''Finds the first d primes'''
    primes = []
    n = 2
    while len(primes) < d:
        # check if 'number' is prime
        is_prime = True
        for prime in primes:
            if n % prime == 0:
                is_prime = False
                break
        # If it is prime, add to the list
        if is_prime:
            primes.append(n)
        n += 1
    return primes

def all_orthogonal(d, bell_func=bell_s, odd_p=False, prop_c=False):
    '''Check if all bell states in given construction are orthogonal.

    Params:
        :d: dimension of the system
        :bell_func: function to generate the bell states. default is bell_s, which generates the bell states in symmetrized way
        :odd_p: whether to use odd only p or normal 0 -> d-1 p. default is False, which uses normal p
        :prop_c: whether to use c = 0, 1, ..., d-1 or just c = n * d//2. default is False, which uses normal c

    '''

    if odd_p:
        p_ls = np.array([2*n+1 for n in range(d)])+2
        # p_ls = np.array([2*n+1 for n in range(d)])+4
        # remove 3 and add one more odd
        # p_ls.remove(3)
        # p_ls.append(2*d+1)
        # what if prime?
        # get the first d primes
        # p_ls = get_d_primes(d)

    else:
        p_ls = range(d)

    if prop_c:
        c_ls = [n*d//2 for n in range(d)]
        # c_ls = [n for n in range(d**2)]
        # remove 3
        # c_ls.remove(3)
        # add one more
        # c_ls.append(2*d+1)

    else:
        c_ls = range(d)

    # initialize list of ((c,p), bell state) pairs
    bell_ls = []

    for c in c_ls:
        for p in p_ls:
            bell = bell_func(d, c, p)
            # check if dot product is 0 with all other bell states
            for cp, b in bell_ls:
                # take dot product
                dot = (b.conj().T @  bell)[0][0]
                if not np.isclose(dot, 0, 1e-10):
                    print('Not orthogonal!')
                    print(dot)
                    c_b = cp[0]
                    p_b = cp[1]
                    print(f'{c_b, p_b}: {b}')
                    print(f'{c, p}: {bell}')
                    return False
            bell_ls.append(((c, p), bell))
    print('All orthogonal!')
    return True

def all_orthogonal3(d, bell_func=bell_s3):
    '''Check if all bell states in given construction are orthogonal using new convention of only correlation class, same logic as above.'''

    c_ls = range(d**2)
    bell_ls = []
    
    for c in c_ls:
        bell = bell_func(d, c)
        # check if dot product is 0 with all other bell states
        for j, b in enumerate(bell_ls):
            # take dot product
            dot = np.dot(b.conj().T, bell)[0][0]
            if not np.isclose(dot, 0, 1e-10):
                print('Not orthogonal!')
                print(dot)
                print(f'c: {j}, {b}')
                print(f'{c}: {bell}')
                return False
        bell_ls.append(bell)

def all_symmetric(d, bell_func=bell_s, odd_p=False, prop_c=False):
    '''Check if all bell states in given construction are symmetric wrt particle exchange.

    Params:
        :d: dimension of the system
        :bell_func: function to generate the bell states. default is bell_s, which generates the bell states in symmetrized way
        :odd_p: whether to use odd only p or normal 0 -> d-1 p. default is False, which uses normal p
        :prop_c: whether to use c = 0, 1, ..., d-1 or just c = n * d//2. default is False, which uses normal c    
    
    '''

    if odd_p:
        p_ls = [2*n+1 for n in range(d)]
        # p_ls = np.array([2*n+1 for n in range(d)])+4
        # remove 3 and add one more odd
        # p_ls.remove(3)
        # p_ls.append(2*d+1)
    else:
        p_ls = range(d)
    
    if prop_c:
        c_ls = [n*d//2 for n in range(d)]
    else:
        c_ls = range(d)

    for c in c_ls:
        for p in p_ls:
            bell = bell_func(d, c, p)
            if not symmetric(bell):
                print('Not symmetric!')
                print(f'{c, p}: {bell}')
                return False
    print('All symmetric!')
    return True

def is_valid_sym_bell_basis(d, bell_func=bell_s, odd_p=False, prop_c=False):
    '''Checks if the bell states generated by bell_func form a valid bell basis symmetric wrt particle exchange for the d^2 dimensional space: that is, symmetric wrt particle exchange, orthogonal, and all states are maximally entangled.

    Params:
        :d: dimension of the system
        :bell_func: function to generate the bell states. default is bell_s, which generates the bell states in symmetrized way
        :odd_p: whether to use odd only p or normal 0 -> d-1 p. default is False, which uses normal p
        :prop_c: whether to use c = 0, 1, ..., d-1 or just c = n * d//2. default is False, which uses normal c
    
    '''    
    if not check_all_entangled(d, bell_gen_func=bell_func, odd_p = odd_p, prop_c = prop_c):
        print('Not all entangled!')
        return False
    
     # check if all bell states are orthogonal
    if not all_orthogonal(d, bell_func=bell_func, odd_p=odd_p, prop_c=prop_c):
        print('Not orthogonal!')
        return False
    
    if not all_symmetric(d, bell_func=bell_func, odd_p = odd_p, prop_c = prop_c):
        print('Not symmetric!')
        return False
    
    
    
    print('All good!')
    return True


if __name__ == '__main__':
    d = 6

    # print(np.nonzero(bell_s(d, 7, 1)))

    # is_valid_sym_bell_basis(d, bell_func=bell_s, odd_p=True, prop_c=True)
    print(display_bell(bell_s(d, 3, 5)))
    print('------')
    print(display_bell(bell_s(d, 3, 3)))
    print('------')
    print(bell_s(d, 3, 5).conj().T @ bell_s(d, 3, 3))



    # factor = max([2*n+1 for n in range(d)])
    # x_ls = np.linspace(0, 1, 100)
    # fig, ax = plt.subplots(2, 1)
    # for p in [n for n in range(d)]:
    #     print(f'--------p = {p}-------')
    #     p_ls= []
    #     for j in range(d):
    #         # print(np.exp(2*np.pi*1j*(2*j+1)*p / factor))
    #         # print(np.exp(2*np.pi*1j*j*p / d))
    #         # p_ls.append(np.exp(2*np.pi*1j*j*p / d))
            
    #         ax[0].plot(x_ls, np.sin(2*np.pi*(2*j+1) * p / factor *x_ls))
    #         ax[1].plot(x_ls, np.sin(2*np.pi*j*p / d *x_ls))
    #         # p_ls.append(np.exp(np.pi*1j*(2*j+1)*p / factor))
    #     # plt.plot(p_ls, label=f'p = {p}')
    # ax[0].legend()
    # ax[1].legend()
    # plt.savefig('figs/p_ls_n.pdf')
    # plt.show()

    
    # for c in range(d**2):
    #     bell = bell_s3(d, c)
    #     print(f'c = {c}, entanglement = {check_entangled(bell, ret_val=True)}')
    #     print(symmetric(bell))

    # all_orthogonal3(d)
    
    # for w in np.linspace(0, 2, 10):
    #     print('-----')
    #     print(f'w = {w}')    
    #     bell_func = partial(bell_s, w=w)
    #     is_valid_sym_bell_basis(d, bell_func=bell_s, odd_p=True, prop_c=True)
    # print(display_bell(bell_s(d, 0, 5)))
    # print('--------')
    # print(display_bell(bell_s(d, 0, 11)))



    # c1, p1 = 1, 2
    # c2, p2 = 2, 3
    # bell1 = bell_s(d, c1, p1)
    # bell2 = bell_s(d, c2, p2)

    # print_matrix(np.round(np.outer(bell1, np.conj(bell1)), 10), title=f'{c1}_{p1}_{d}')
    # print('------')
    # print_matrix(np.round(np.outer(bell2, np.conj(bell2)), 10), title=f'{c2}_{p2}_{d}')

    # bell = bell_s(d=d, c=9, p=3)
    # print('entanglement:', check_entangled(bell, ret_val=True))

    # all_orthogonal(d, bell_func=bell_s, odd_p=True, even_c = True)
    # check_all_entangled(d, bell_gen_func=bell_s, odd_p=True, even_c = True)

    



    # bell = make_bell(d=d, c=3, p=0)
    # print('entanglement:', check_entangled(bell, ret_val=True))
    # bell_str = display_bell(bell)
    # print(bell_str)
    # print('------')

    # bell_add = '1 |0>|1> + 1 |1>|0>'
    # bell_tot = correct_bell(bell, bell_add)
    # print('entanglement:', check_entangled(bell_tot, ret_val=True))
    # print(display_bell(bell_tot))

    


    # 2.178436944154104e-07
    # x_best = np.array([1.1997766718739271, 0.2907492336982105, 0.39058612525764613, -0.25800554800457315, 0.2907189484204616, 0.9062780259041427, -0.6162903349319176, 0.5123047254690065, 0.3905610806086528, -0.6163054806422082, -0.041125496512232865, 0.589393242091415, -0.2579983973239351, 0.5123126946718567, 0.589415047039367, -0.15156466406436406, 0.16378271096375777, -0.29319488806431654, 0.168736597332674, -0.33205964778452635, -0.2932049809147356, 0.10345587141750738, -0.28618211804482585, 0.3082520017570542, 0.16874070004525968, -0.28618116089140777, 0.34476755031744744, 0.8472527206234807, -0.3320591801345535, 0.30825506534625174, 0.8472623994986536, 0.44267075823579843])

    # print(f'loss: {loss(x_best, bell)}')
    
    # x_best = make_entangled(bell, x0_ls=[x_best])
    
    # corr = x_best[:d**2]+1j*x_best[d**2:]
    # corr = corr.reshape((d**2, 1))
    # corr_str = display_bell(corr)
    # bell_corr = bell + corr
    # bell_corr /= np.linalg.norm(bell_corr)
    # print('actual xbest', check_entangled(bell_corr, display_val=True))



    # coeff_str = '0.4432427419467506e^0.135672382684276841j |0>|0>+ 0.15114436852118282e^-0.78958630396705731j |0>|1>+ 0.15574231179815554e^0.407792055538780731j |0>|2>+ 0.15392544651512838e^-2.2313456871371841j |0>|3>+ 0.15113918626738287e^-0.7896555971982711j |1>|0>+ 0.3338910271548267e^0.113662653137927031j |1>|1>+ 0.24872439401574603e^-2.7068592748368281j |1>|2>+ 0.21885424056587785e^0.54166605263735571j |1>|3>+ 0.15573449178174745e^0.40782425306561061j |2>|0>+ 0.24872927476010348e^-2.7068699399226141j |2>|1>+ 0.1270943457055362e^1.68952005508808181j |2>|2>+ 0.3777912053378135e^0.96299429478083921j |2>|3>+ 0.1539237054129368e^-2.23133294147367561j |3>|0>+ 0.21885731821629087e^0.54166357129468391j |3>|1>+ 0.3777986717123001e^0.96298230724144271j |3>|2>+ 0.17127097129382063e^1.90067283193149831j |3>|3>'

    # coeffs = coeff_str.split('+') # evaluate each term
    # coeff_tot = convert_bell_str(coeff_str, d)
    # for coeff in coeffs:
    #     print(coeff)
    #     coeff_vec = convert_bell_str(coeff, d)
    #     bell_cor = bell + coeff_vec
    #     bell_cor /= np.linalg.norm(bell_cor)
    #     print(check_entangled(bell_cor, display_val=True))
    #     print('-----')

   
    # print(coeff_tot)
    # print('*~*~*~')
    # print(corr)
    # print('*~*~*~')
    # print(f'norm = {np.linalg.norm(coeff_tot - corr)}')

   


    # print(check_entangled(bell_s(d, 3, 2), display_val=True))
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

    #     x_best =  np.array([
#     -0.39914514+0.j, 0.66802457+0.j, 0.37203545+0.j, -0.04055752+0.j, 
#     0.16172204+0.j, -0.25811277+0.j, 0.59908882+0.j, -0.31932903+0.j, 
#     -0.0760946 +0.j, -0.37229527+0.j, 0.40606246+0.j, 0.65354633+0.j, 
#     0.68968878+0.j, 0.2965142 +0.j, 0.08393382+0.j, 0.16422431+0.j, 
#     -0.03314695+0.21097571j, -0.00292451+0.03471294j, -0.02812834+0.34831972j, 
#     -0.02904904+0.53232796j, -0.02018416+0.15073022j, -0.03323988+0.31055887j, 
#     -0.02743774+0.55702722j, -0.03312472+0.05832004j, -0.04125797+0.07318349j, 
#     -0.03802235+0.56537948j, -0.05016439+0.37928723j, -0.04993331+0.42466573j, 
#     -0.0270885 +0.10596727j, -0.06731255+0.27162313j, -0.06944027+0.47743037j, 
#     -0.05154533+0.3501594j
# ]) # loss = 3.48e-07; not symmetric


    # found one to 1e-6!
    # 1.439455219012125e-06
    # x_best = np.array([ 
    #     1.33466454+0.j, 0.07991565+0.j, 1.02935334+0.j, 0.0366273 +0.j, 
    #     0.05358747+0.j, 1.49809086+0.j, 0.41675159+0.j, 0.13737118+0.j, 
    #     1.00378665+0.j, 0.413834 +0.j, -1.13016502+0.j, 1.12790242+0.j, 
    #     0.04500242+0.j, 0.1557045 +0.j, 1.15604772+0.j, 0.88953439+0.j, 
    #     0.13977298+0.56855612j, 0.04592478+0.35555318j, 0.03116871+0.36767168j, 
    #     0.02682425+0.32866123j, 0.04411716+0.84818717j, 0.03743956+0.5903136j, 
    #     0.03449962+0.82848295j, 0.03551454+0.09278652j, 0.03277864+0.52779357j, 
    #     0.03484632+0.4725063j, 0.03543179+0.3985269j, 0.03604953+0.68849192j, 
    #     0.02781067+0.32030142j, 0.03556816+0.36482586j, 0.03580318+0.58050521j, 
    #     0.03926356+0.7201627j 
    # ])