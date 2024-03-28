# testing LU equivalence of correlation pairs
import numpy as np

def shift_standard_L_R(d, n, m, vec_shift=False):
    ''''d is dimension of bell state, n is shift to L, m is shift to R'''
    if not vec_shift:
        for c in range(d):
            print(f'c = {c}')
            for i in range(d):
                L_o = (i) % d
                R_o = (i+c) % d
                L_n = (i + n) % d
                R_n = (i + m+c) % d
                print(f'{L_o}{R_o} --> {L_n}{R_n}')
    else:
        # vecL = np.random.permutation(np.arange(d))
        # vecR = np.random.permutation(np.arange(d))
        vecL= np.arange(d)
        vecR= np.arange(d)
        for c in range(d):
            print(f'c = {c}')
            for i in range(d):
                L_o = i % d
                R_o = (i + c) % d
                L_n = (i + vecL[i] + c) % d
                R_n = (i + vecR[i] + c) % d
                print(f'{L_o}{R_o} --> {L_n}{R_n}')


if __name__ == '__main__':
    shift_standard_L_R(6, 0, 2)