# file to try out Lynn's idea of symmetrized bell states

import numpy as np

def bell_old(d, c, p):
    bell = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        L = j
        R = (j+c) % d
        index = L*d + R
        bell[index] = np.exp(2*np.pi*1j*p / d)

    # normalize
    bell /= np.sqrt(d)

    return bell


def symmetric(bell=None, display=False):
    '''Checks if an input bell state (in joint particle basis) is symmetric wrt particle exchange.

    Params:
        bell: d^2 x 1 vector
        display: boolean to display initial and transformed

    Note: by construction, the L state corresponds to the index // d, R is index % d. Thus, index = L*d + R
    '''
    d = np.sqrt(len(bell))

    # need to swap L and R and check if they are the same
    new_bell = np.zeros((int(d**2),1), dtype=complex)
    for i, b in enumerate(bell):
        b = b[0]
        if b != 0:
            L = i // d
            R = i % d
            new_i = int(R*d + L)
            new_bell[new_i] = b
    
    if display:
        print(bell)
        print('------')
        print(new_bell)
        return np.all(np.isclose(bell, new_bell, 1e-10))

if __name__ == '__main__':
    print(symmetric(bell_old(d=4, c=0, p=2)))