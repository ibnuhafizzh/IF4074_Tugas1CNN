from operator import le
from layer import *
from generateImage import *
from sklearn.model_selection import train_test_split, KFold
from model import *

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
    # dummy_matrix = matrix_images[20]

    # for layer in Layers:
    #     dummy_matrix = layer.forward(dummy_matrix)
    # print("Picture: ",folder_path[20])
    # print("Predict Result :", class_dictionary[str(int(dummy_matrix[0]))])

    X_train, X_test, y_train, y_test = train_test_split(matrix_images, class_label, test_size=0.1)

    cnn = Model(Layers)
    cnn.fit(features=X_train, target=y_train, batch_size=5, epoch=5, learn_rate=0.1)

    filename = "model1"
    cnn.save_model(filename)

    cnn2 = Model()
    cnn2.load_model(filename)

    output = np.array([])
    for data in X_test:
        for layer in cnn2.layers:
            forward_cnn = layer.forward(data)
        output = np.append(output, np.rint(forward_cnn))
    
    print("\nPredicted:", output)
    print("\nAccuracy:", metrics.accuracy_score(y_test, output))
    print("\nConfusion matrix:\n", metrics.confusion_matrix(y_test, output))
