import numpy as np

def relu(input):
    return np.max(input,0)

def sigmoid(input):
    return 1/(1+np.exp(-input))
