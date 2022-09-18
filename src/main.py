from layer import *
from generateImage import *

DATA_PATH = "./test"

if __name__ == "__main__":
    dataset_path = DATA_PATH
    folder_path, class_label, class_dictionary = read_dataset(dataset_path)
    Layers = [
        ConvolutionLayer(3,10,2,0,2),
        Pooling(2,2,"max"),
        FlattenLayer(),
        DenseLayer(2,"relu"),
        DenseLayer(4,"relu"),
        DenseLayer(1,"sigmoid")
    ]

    matrix_images = list_img_to_matrix(folder_path)
    dummy_matrix = matrix_images[0]

    for layer in Layers:
        dummy_matrix = layer.forward(dummy_matrix)
    print("Predict Result :", class_dictionary[str(int(dummy_matrix[0]))])