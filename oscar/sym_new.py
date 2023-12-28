# file to test new symmetrization method
import numpy as np
import math
import matplotlib.pyplot as plt
from symmetrized import check_entangled, symmetric, display_bell, convert_bell_str
from oscars_toolbox.trabbit import trabbit

def permutation_to_integer(perm):
    '''Converts a permutation to its integer value via reverse CNS.

    Algorithm: counts num of elements smaller than each element to the right; multiplies by the factorial of the index of the element.
    
    '''
    n = len(perm)
    val = 0
    for i in range(n):
        # count num elements are smaller than perm[i] to the right
        smaller_count = sum(perm[j] < perm[i] for j in range(i + 1, n))

        # add the contribution of this element to the total integer value
        val += smaller_count * math.factorial(i)

    return val

def bell_6(c, p, add_sec_term=True):
    '''Generates d = 6 bell states for a given c (new definition from before) and p. assumes indistinguishable particles.

    Params:
        c (int): correlation class, bounded by 0 and 5
        p (int): phase class, bounded by 0 and 5
        add_sec_term (bool): whether to add the second term in the bell state (see notes)

    '''

    RL_ls_ls = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], [(0, 1), (2, 3), (4, 5)], [(0, 2), (1, 4), (3, 5)], [(0, 3), (2, 4), (1, 5)], [(0, 4), (1, 3), (2, 5)], [(0, 5), (1, 2), (3, 4)]]
    
    RL_ls = RL_ls_ls[c]

    # initialize bell vector
    if add_sec_term:
        bell_vec = np.zeros((36, 1), dtype=complex)
    else:
        bell_vec = np.zeros((12, 1), dtype=complex)
    for i in range(len(RL_ls)): 
        L, R = RL_ls[i]
        # get the index of the permuted bell state
        index1 = L + R

        # phase = np.exp(2 * np.pi * 1j * i * p / 3)
        if len(RL_ls) == 3: # 3 terms * 2 for complex = 6 params
            phase = np.exp(2 * np.pi * 1j * i * p / 3)

        else: # c = 0 term
            phase = np.exp(2 * np.pi * 1j * i * p / 6)

        if add_sec_term:
            index1 = 6*L + R
            index2 = 6*R + L
            bell_vec[index2] = phase
         # get the bell state
        bell_vec[index1] = phase

    return bell_vec

def check_all_entangled():
    '''Checks all bell states for entanglement.'''
    for c in range(6):
        if c == 0:
            for p in range(6):
                bell = bell_6(c, p)
                if not check_entangled(bell, display_val=False):
                    print(f'c = {c}, p = {p} is not entangled.')
                    return False
        else:
            for p in range(3):
                bell = bell_6(c, p)
                if not check_entangled(bell, display_val=False):
                    print(f'c = {c}, p = {p} is not entangled.')
                    return False
    print('All bell states are entangled.')
    return True

def check_all_symmetric():
    '''Checks all bell states for symmetry.'''
    for c in range(6):
        if c == 0:
            for p in range(6):
                bell = bell_6(c, p)
                if not symmetric(bell):
                    print(f'c = {c}, p = {p} is not symmetric.')
                    return False
        else:
            for p in range(3):
                bell = bell_6(c, p)
                if not symmetric(bell):
                    print(f'c = {c}, p = {p} is not symmetric.')
                    return False
            
    print('All bell states are symmetric.')
    return True

def check_all_orthogonal():
    '''Checks all bell states for orthogonality.'''
    def check_orthogonal(c1, p1, c2, p2):
        bell1 = bell_6(c1, p1)
        bell2 = bell_6(c2, p2)
        if np.abs(bell1.conj().T @ bell2) > 1e-10:
            print(f'({c1}, {p1}) and ({c2}, {p2}) are not orthogonal.')
            display_bell(bell1)
            print('------')
            display_bell(bell2)
            print(bell1.conj().T @ bell2)
            return False
        return True
    # c can take on vals 0 to 6
    # p can take on 0 to 6 if c = 0, 0 to 3 otherwise
    for c1 in range(6):
        if c1 == 0:
            for p1 in range(6):
                for c2 in range(c1, 6):
                    if c2 == c1:
                        for p2 in range(p1+1, 6):
                            if not check_orthogonal(c1, p1, c2, p2):
                                return False
                    else:
                        if c2 == 0:
                            for p2 in range(3):
                                if not check_orthogonal(c1, p1, c2, p2):
                                    return False
                        else:
                            for p2 in range(6):
                                if not check_orthogonal(c1, p1, c2, p2):
                                    return False
        else:
            for p1 in range(3):
                for c2 in range(c1, 6):
                    if c2 == c1:
                        for p2 in range(p1+1, 3):
                            if not check_orthogonal(c1, p1, c2, p2):
                                return False
                    else:
                        if c2 == 0:
                            for p2 in range(3):
                                if not check_orthogonal(c1, p1, c2, p2):
                                    return False
                        else:
                            for p2 in range(6):
                                if not check_orthogonal(c1, p1, c2, p2):
                                    return False
                                
    print('All bell states are orthogonal.')
    return True

def loss_phase(params):
    '''Loss function for phases.'''
    # unpack the phase_ls, which puts all real and then all imaginary parts together
    # real_ls = phase_ls[:3]
    # imag_ls = phase_ls[3:]
    # phase_ls = [real_ls[i] + 1j * imag_ls[i] for i in range(len(real_ls))]
    loss = 0
    c = 1
    for p in range(6):
        phase_ls1 = [np.exp(2*np.pi*1j*params[0]*p*i + 2*np.pi*1j*params[1]*i) for i in range(3)]
        bell = bell_6(c, p, phase_ls1)
        for p2 in range(6):
            if p != p2:
                phase_ls2 = [np.exp(2*np.pi*1j*params[0]*p2*i + 2*np.pi*1j*params[1]*i) for i in range(3)]
                bell2 = bell_6(c, p2, phase_ls2)
                loss += np.abs(bell.conj().T @ bell2)**2
        # loss += (np.linalg.norm(phase_ls) - 1)**2
    return loss

def find_phases():
    '''Finds the phase factors for c > 0 bell states'''
    # initialize the phases
    def random_phases():
        # phase_ls = []
        # for _ in range(6):
        #     phase_ls.append(np.random.random())
        # # need to normalize by norm of complex
        # phase_ls = np.array(phase_ls)
        # phase_ls = phase_ls[:3] + 1j*phase_ls[3:]
        # phase_ls /= np.linalg.norm(phase_ls)
        # phase_real_ls = [phase.real for phase in phase_ls]
        # phase_imag_ls = [phase.imag for phase in phase_ls]
        # return phase_real_ls + phase_imag_ls
        return np.random.random(2)
    
    x_best, loss_best = trabbit(loss_phase, random_phases)
    print(list(x_best))
    print(loss_best)
    return x_best


if __name__ == '__main__':
    # check_all_entangled()
    # check_all_symmetric()
    # check_all_orthogonal()
    # bell = bell_6(1, 0, add_sec_term=False)
    # display_bell(bell)
    # print(check_entangled(bell, display_val=True, add_sec_term=False))
    # print(symmetric(bell))

    N = lambda n: 2**(2*n-1)-1
    k = lambda n: 2**(n+1)-1
    plt.plot([N(n) for n in range(1, 10)], label='N')
    plt.plot([k(n) for n in range(1, 10)], label='k')
    plt.plot([2**(2*n) for n in range(1, 10)], label='n^2')
    plt.legend()
    plt.show()

    





    

    # for p in range(6):
    #     plt.plot([np.exp(2*np.pi*1j*i*(p%3)/3 + 2*np.pi*1j*i/(6+p)) for i in range(6)], label=f'p = {p}')
    #     # plt.plot([np.exp(2*np.pi*1j*i*p/6) for i in range(6)], label=f'p = {p}')

    # plt.legend()
    # plt.show()


    # perm0 = [0, 1, 2, 3, 4, 5]
    # perm1 = [1, 0, 3, 2, 5, 4]
    # perm2 = [2, 4, 0, 5, 1, 3]
    # perm3 = [3, 5, 4, 0, 2, 1]
    # perm4 = [4, 3, 5, 1, 0, 2]
    # perm5 = [5, 2, 1, 4, 3, 0]
    # perm_ls = [perm0, perm1, perm2, perm3, perm4, perm5]
    # for perm in perm_ls:
    #     perm_str = ''
    #     for i in range(len(perm)):
    #         if i == 0:
    #             perm_str += '|'+str(perm0[i])+str(perm[i])+'>'
    #         else:
    #             perm_str += '+ |'+str(perm0[i])+str(perm[i])+'>'
    #     print(perm_str)
   