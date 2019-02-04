import numpy as np
import sys

'''
sys.argv[1] => value
sys.argv[2] => mean
sys.argv[3] => std
'''

def pdf(x, mean, std):
  return 1/(std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * std**2))

    
x_ = float(sys.argv[1])
mean_ = float(sys.argv[2])
std_ = float(sys.argv[3])

print pdf(x_, mean_, std_)
