# code to try out the odd phase class idea
import numpy as np
from oscars_toolbox.trabbit import trabbit
from functools import partial

def get_sum(beta):
    '''Takes as input real beta and computes the value of alpha and then sees if e^{2pi i alpha} + e^{2pi i (alpha + beta)} equals -1'''

    # get alpha
    alpha = 1/2 + 1j*np.log(1+np.exp(1j*2*np.pi*beta))/(2*np.pi)

    print(f'alpha = {alpha}')

    sum = np.exp(2*np.pi*1j*alpha) + np.exp(2*np.pi*1j*(alpha + beta))
    print(f'sum = {sum}')

def get_alpha(px):
    '''Takes as input px and py and computes the value of alpha'''
    # get beta
    # beta = px - py
    beta=px

    # get alpha
    return 1/2 + 1j*np.log(1+np.exp(1j*2*np.pi*beta))/(2*np.pi)

def test_3(params):
    '''test of logic for d = 3'''

    # p1, p3, p5 = params
    # p0_real, p1_real, p2_real, p3_real, p4_real, p5_real = params[:6]
    # p0_imag, p1_imag, p2_imag, p3_imag, p4_imag, p5_imag = params[6:]
    
    # p0, p1, p2, p3, p4, p5 = params
    # p0 = 0
    # p2 = 0
    # p4 = 0

    p0, p1, p2, p3 = params
    


    # # get alpha values
    # alpha1 = get_alpha(p1)
    # alpha3 = get_alpha(p3)
    # alpha5 = get_alpha(p5)

    # # get other p vals
    # p2 = 0
    # p4 = alpha5
    # p0 = 1/2*(alpha1 + alpha3 - p2 - p4)

    print(np.exp(2*np.pi*1j*p1) + np.exp(2*np.pi*1j*p3))
    print(np.exp(2*np.pi*1j*(p0 + p1)) + np.exp(2*np.pi*1j*(p2 + p3)))


    # create vectors
    v0 = np.array([1, 1, 1])
    v1 = np.array([1, np.exp(2*np.pi*1j*p1), np.exp(2*np.pi*1j*(p0 + p1))])
    v2 = np.array([1, np.exp(2*np.pi*1j*p3), np.exp(2*np.pi*1j*(p2 + p3))])

    # compute inner products
    ip01 = v0 @ v1.conj().T
    ip12 = v1 @ v2.conj().T
    ip20 = v2 @ v0.conj().T

    # print(f'ip01 = {ip01}')
    # print(f'ip12 = {ip12}')
    # print(f'ip20 = {ip20}')

    return np.abs(ip01) + np.abs(ip12) + np.abs(ip20)

# write test3 in sympy
import sympy as sp

def test_3_sympy(vals=None):
    '''test of logic for d = 3'''

    params = sp.symbols('p0 p1 p2 p3 p4 p5', real=True)

    if vals is not None:
        # get list of indices to substitute
        indices = [pair[1] for pair in vals]
        vals = [pair[0] for pair in vals]
        test_params = []
        for i in range(len(params)):
            if i in indices:
                test_params.append(vals[indices.index(i)])
            else:
                test_params.append(params[i])

    p0, p1, p2, p3, p4, p5 = params

    # create vectors
    v0 = sp.Matrix([[1], [sp.exp(2*sp.pi*sp.I*p1)], [sp.exp(2*sp.pi*sp.I*(p0 + p1))]])
    v1 = sp.Matrix([[1], [sp.exp(2*sp.pi*sp.I*p3)], [sp.exp(2*sp.pi*sp.I*(p2 + p3))]])
    v2 = sp.Matrix([[1], [sp.exp(2*sp.pi*sp.I*p5)], [sp.exp(2*sp.pi*sp.I*(p4 + p5))]])

    # compute inner products
    ip01 = v1.conjugate().T @ v0
    ip12 = v2.conjugate().T @ v1
    ip20 = v0.conjugate().T @ v2

    eq0 = sp.exp(2*sp.pi*sp.I*p1) + sp.exp(2*sp.pi*sp.I*p3) + sp.exp(2*sp.pi*sp.I*p5)
    eq1 = sp.exp(2*sp.pi*sp.I*(p0 + p1)) + sp.exp(2*sp.pi*sp.I*(p2 + p3)) + sp.exp(2*sp.pi*sp.I*(p4 + p5))

    # print('ip01 = ')
    # sp.pprint(ip01)
    # print('ip12 = ')
    # sp.pprint(ip12)
    # print('ip20 = ')
    # sp.pprint(ip20)

    results= sp.Matrix([ip01, ip12, ip20])
    eqs = sp.Matrix([eq0, eq1])
    if vals is not None:
        results = results.subs({params[i]: test_params[i] for i in range(len(params))}) 
        eqs = eqs.subs({params[i]: test_params[i] for i in range(len(params))})

    for result in results:
        sp.pprint(sp.N(result))
    print('------')
    for eq in eqs:
        sp.pprint(sp.N(eq))

    # return sp.Abs(ip01) + sp.Abs(ip12) + sp.Abs(ip20)

def gen_random_params_3():
    # find random p1, p3, p5
    # return np.random.rand(6)
    return np.random.rand(6)

def find_params_3():
    '''Finds the parameters for d = 3'''
    # loss = partial(test_3, )

    # find random p1, p3, p5
    x_best, loss_best = trabbit(test_3, gen_random_params_3, alpha=0.8, tol=1e-7)
    print(f'x_best = {x_best}')
    print(f'loss_best = {loss_best}')

    # first 3 are real, last 3 are imaginary. only dealing with p0, p2, p4
    p0_real, p2_real, p4_real = x_best[:3]
    p0_imag, p2_imag, p4_imag = x_best[3:6]

    p0 = p0_real + 1j*p0_imag
    p2 = p2_real + 1j*p2_imag
    p4 = p4_real + 1j*p4_imag


    # print out differences between each terms
    for i in range(len(x_best)):
        for j in range(i+1, len(x_best)):
            print(f'x_{i} - x_{j} = {x_best[i] - x_best[j]}')
        

    return x_best

def get_DFT(d):
    n = np.arange(d)
    k = n.reshape((d, 1))
    omega = k * n / (d)

    #print sum of first row
    print(sum([np.exp(2*np.pi*1j*o) for o in omega[1][1:]]))
    print(sum([-np.exp(2*np.pi*1j*o) for o in omega[1][1:]]))

    # remove first row and column
    # omega = omega[1:, 1:]
    return omega


if __name__ == '__main__':
    # test_3()
    # find_params_3()
    # print(get_DFT(3))

    # test_3_sympy([(0,0), (1/2,1), (1, 3), (3/2, 5), (0, 2), (0, 4)])
    # test_3_sympy([ (1/2,0), (-0.06087815,1) , (1/2,2)  ,(0.60578852,3), (0,4),  (0.27245519, 5)])
    # test_3_sympy([(1/2, 0), (1/2, 2), (1/2, 4)])

    # print(sum(np.exp(2*np.pi*1j*np.array([2/3+1/2, 1, 1]))))

    # print(test_3([0, 1/3, 0, 2/3]))
    # print(sum([np.exp(2*np.pi*1j/2), np.exp(2*np.pi*1j*2/2), np.exp(2*np.pi*1j*3/2)]))
    # print(sum([np.exp(2*np.pi*1j*1/2), np.exp(2*np.pi*1j*0)]))

    v0 = np.array([1, 1, 1])
    v1 = np.array([1, np.exp(2*np.pi*1j*1/3), np.exp(2*np.pi*1j*(2/3))])
    v2 = np.array([1, np.exp(2*np.pi*1j*(2/3)), np.exp(2*np.pi*1j*( 4/3))])
    print(v0 @ v1.conj().T)
    print(v1 @ v2.conj().T)