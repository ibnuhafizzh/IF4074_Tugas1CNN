from layer import *
import numpy as np

if __name__ == "__main__":
    Layers = [
        ConvolutionLayer(3,10,2,0,2),
        Pooling(2,2,"max"),
        FlattenLayer(),
        DenseLayer(2,"relu"),
        DenseLayer(4,"relu"),
        DenseLayer(1,"sigmoid")
    ]
    matrix = np.random.randn(3,9,9)
    print(matrix)
    for layer in Layers:
        matrix = layer.forward(matrix)
    print(matrix)