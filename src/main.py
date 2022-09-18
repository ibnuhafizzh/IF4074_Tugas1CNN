from layer import *
import numpy as np
from generateImage import *

DATA = "/Users/mac/Documents/ITB/Semester7/ML/IF4074_Tugas1CNN/test"

if __name__ == "__main__":
    dataset_path = DATA
    file_path, class_label, class_dictionary = read_dataset(dataset_path)
    Layers = [
        ConvolutionLayer(3,10,2,0,2),
        Pooling(2,2,"max"),
        FlattenLayer(),
        DenseLayer(2,"relu"),
        DenseLayer(4,"relu"),
        DenseLayer(1,"sigmoid")
    ]
    matrix = list_img_to_matrix(file_path)[1]
    print(matrix)
    for layer in Layers:
        matrix = layer.forward(matrix)
    print(matrix)