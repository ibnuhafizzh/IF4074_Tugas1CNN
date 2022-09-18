from multiprocessing import pool
from turtle import forward
import numpy as np
import common
from tensorflow.keras import layers # for test
import tensorflow as tf

class DenseLayer:
    def __init__(self, n_unit, activation):
        self.n_unit = n_unit
        self.activation = activation
        self.bias = np.zeros(n_unit)
        self.weight = np.random.rand(n_unit)

    def forward(self,inputs):
        multisum = np.array([])
        for i in range(self.n_unit):
            multisum = np.append(multisum, 
            np.sum(np.multiply(self.weight[i], inputs)) + self.bias[i])

        if self.activation == 'sigmoid':
            matrixsigmoid = np.vectorize(common.sigmoid)
            return matrixsigmoid(multisum)
        else:
            return np.maximum(multisum, 0)

class ConvolutionLayer:
    def __init__(self) -> None:
        pass

    def detector(self,input):
        return common.relu(input)

class Pooling:
    def __init__(self,filter_size,stride_size,mode):
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.mode = mode

    def forward(self, inputs):
        self.input = inputs
        depth = inputs.shape[0]
        length = ((inputs.shape[1] - self.filter_size) // self.stride_size) + 1
        pooled_map = np.zeros([depth, length, length], dtype=np.double)

        for d in range(0, depth):
            for w in range(0, length):
                for h in range(0, length):
                    if (self.mode.lower() == 'average'):
                        pooled_map[d,w,h] = self.average(inputs,d,w,h)
                    elif (self.mode.lower() == 'max'):
                        pooled_map[d,w,h] = self.max(inputs,d,w,h)
        return pooled_map
    
    def average(self,inputs,d,r_pos,b_pos):
        return np.average(inputs[d, 
                                r_pos*self.filter_size:(r_pos*self.filter_size + self.filter_size),
                                b_pos*self.filter_size:(b_pos*self.filter_size + self.filter_size)
                                ])

    def max(self,inputs,d,r_pos,b_pos):
        return np.max(inputs[d, 
                                r_pos*self.filter_size:(r_pos*self.filter_size + self.filter_size),
                                b_pos*self.filter_size:(b_pos*self.filter_size + self.filter_size)
                                ])

class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        output = inputs.flatten()
        return output

if __name__ == "__main__":
    # for test
    test = []
    arr1 = np.array([[[1,1,2,4]
            ,[5,6,7,8]
            ,[3,2,1,0]
            ,[1,2,3,4]]])
    pool = Pooling(2,2,"max")
    print(pool.forward(arr1))