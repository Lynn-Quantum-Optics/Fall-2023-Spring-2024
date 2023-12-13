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
    def min_func(x):
        '''Function to minimize loss function. Uses nelder-mead algorithm. Returns loss and params.

        Parameters:
            :x: initial guess
            :return_param: whether to return params or not
        
        '''
        result = minimize(loss_func, x)
        return result.x
     
    # try initial guesses #
    if x0_ls is None:
        x0_ls = [random_gen()]

    x_best = None
    loss_best = np.inf

    for x0 in x0_ls:
        x_min = min_func(x0)
        loss = loss_func(x_min)
        if loss < loss_best:
            x_best = x_min
            loss_best = loss

    ## ----- gradient descent ----- ##
    i = 0
    isi = 0 # index since improvement
    try:
        while i < num and loss_best > tol:
            if verbose:
                print(f'iter: {i}, isi: {isi}, current loss: {loss}, best loss: {loss_best}')
            # if we haven't, then hop out and use a new random input
            if isi == int(num * frac):
                if verbose:
                    print('hopping out')
                x_min = random_gen()
                isi=0
            else: # gradient descent
                grad = approx_fprime(x_min, min_func, 1e-8)
                if np.all(grad < tol*np.ones_like(grad)): # if gradient is too small, then hop out
                    x0 = random_gen()
                else:
                    x0 = x0 - alpha*grad
            # now minimize
            x_min = min_func(x_min)
            loss = loss_func(x_min)
            if loss < loss_best:
                x_best = x_min
                loss_best = loss
            else:
                isi += 1 # if no update, then increment isi
            i += 1
    except KeyboardInterrupt:
        print('Exiting...')
        # confirm loss is correct
        print('Loss from x_best: ', loss_func(x_best))
        print('x_best:', list(x_best))

    return x_best, loss_best