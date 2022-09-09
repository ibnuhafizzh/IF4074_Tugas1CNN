from turtle import forward
import numpy as np
import common

class DenseLayer:
    def __init__(self, n_unit, activation, inputs):
        pass

    def calculateoutput(self,inputs):
        res = np.matmul(inputs, self.weights)
        pass

    def forward(self,inputs):
        pass

class ConvolutionLayer:
    def __init__(self) -> None:
        pass

    def detector(self,input):
        common.relu(input)
        pass

class Pooling:
    def __init__(self) -> None:
        pass
