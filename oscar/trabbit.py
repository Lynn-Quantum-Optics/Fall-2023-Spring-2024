# custom gradient descent algorithm; tortoise and rabbit (trabbit)

import numpy as np
from scipy.optimize import minimize, approx_fprime 
from functools import partial

def trabbit(loss_func, random_gen, x0_ls=None, num = 1000, alpha=0.3, frac = 0.1, tol = 1e-5, verbose=False):
    '''Function to implement my custom gradient descent algorithm, trabbit. Goal is to perform double optimization

    Parameters:
    :loss_func: function to minimize. assumes all arguments already passed through partial.
    :random_gen: function to generate random inputs
    :x0_ls: initial guess within a list. if None, then random_gen is used. can also be a list, in which case all initial params will be tried before implementing gd
    :N: number of iterations
    :alpha: learning rate
    :frac: fraction of iterations to use for rabbit (to hop out and use new random input)
    :tol: tolerance for convergence. if loss is less than tol, then stop
    :verbose: whether to print out loss at each iteration

    Returns:
    :x_best: best params
    :loss_best: best loss
    
    '''
    def min_func(x, return_param=True):
        '''Function to minimize loss function. Uses nelder-mead algorithm. Returns loss and params.

        Parameters:
            :x: initial guess
            :return_param: whether to return params or not
        
        '''
        result = minimize(loss_func, x)
        if return_param:
            return result.fun, result.x
        else:
            return result.fun
    
    # try initial guesses #

    if x0_ls is None:
        x0_ls = [random_gen()]

    x_best = None
    loss_best = np.inf

    for x0 in x0_ls:
        loss, x = min_func(x0)
        if loss < loss_best:
            x_best = x
            loss_best = loss

    ## ----- gradient descent ----- ##
    i = 0
    isi = 0 # index since improvement
    while i < num and loss_best > tol:
        if verbose:
            print(f'iter: {i}, isi: {isi}, current loss: {loss}, best loss: {loss_best}')
        # if we haven't, then hop out and use a new random input
        if isi == int(num * frac):
            if verbose:
                print('hopping out')
            x = random_gen()
            isi=0
        else: # gradient descent
            min_func_val = partial(min_func, return_param=False) # only want to consider min func value for gd
            grad = approx_fprime(x, min_func_val, 1e-6)
            if np.all(grad < tol*np.ones_like(grad)): # if gradient is too small, then hop out
                x0 = random_gen()
            else:
                x0 = x0 - alpha*grad
        # now minimize
        loss, x_best = min_func(x)
        if loss < loss_best:
            x_best = x
            loss_best = loss
        else:
            isi += 1 # if no update, then increment isi
        i += 1

    return x_best, loss_best