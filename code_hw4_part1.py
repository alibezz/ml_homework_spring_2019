#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

def poly_function(x):
    return 4*x**4 - 15*x**3 + 11*x**2 + 10*x + 2

xvals = np.linspace(-1,3,1000)
yvals = list(map(poly_function, xvals)) 

plt.plot(xvals, yvals)
plt.savefig('fig_hw4_part1_q1.png')

from scipy import optimize
local_minimum = optimize.brent(poly_function,brack=(2,3))
print 'local minimum', local_minimum, poly_function(local_minimum) 
global_minimum = optimize.basinhopping(poly_function, [1.])
print 'global minimum', global_minimum.x, global_minimum.fun

# Suppose we apply gradient descent to this function, starting with x = −1. To do this, we will
# need to update x using the update rule
# x = x + −y ∗ f'(x)
# where f' is the derivative of f, and y is the "step size".
# Write a small program implementating gradient descent for this function. Setting x = −1 and
# y = 0.001, run gradient descent for 6 iterations (that is, do the update 6 times). Report the
# values of x and f(x) at the start and after each of the first 6 iterations.
# Run the gradient descent again, starting with x=-1, for 1200 iterations. Report the last 6 values
# of x and f(x).
# Has the value of x converged? Has the gradient descent found a minimum? Is it the global or
# the local minimum?
# You do NOT have to hand in your code

def derivative_poly_function(x):
    return 16*x**3 - 45*x**2 + 22*x + 10

def gradient_descent(x, eta, num_iter):
    for i in xrange(num_iter):
        print 'iteration', i
        print 'before', x, poly_function(x)
        x = x - eta*derivative_poly_function(x)
        print 'after', x, poly_function(x)

#gradient_descent(-1, 0.001, 6)
#gradient_descent(-1, 0.001, 1200)

#Repeat the previous exercise, but this time, start with x=3
#gradient_descent(3, 0.001, 6)
#gradient_descent(3, 0.001, 1200)

#gradient_descent(-1, 0.01, 6)
#gradient_descent(-1, 0.01, 1200)

gradient_descent(-1, 0.1, 100)
