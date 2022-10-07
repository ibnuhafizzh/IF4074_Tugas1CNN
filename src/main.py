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

    # X_train, X_test, y_train, y_test = train_test_split(matrix_images, class_label, test_size=0.1)

    # cnn = Model(Layers)
    # cnn.fit(features=X_train, target=y_train, batch_size=5, epoch=3, learn_rate=0.1)

    # filename = "model1"
    # cnn.save_model(filename)

    # cnn2 = Model()
    # cnn2.load_model(filename)

    # output = np.array([])
    # for data in X_test:
    #     forward_cnn = cnn2.forward(data)
    #     output = np.append(output, np.rint(forward_cnn))
    
    # print("\nPredicted:", output)
    # print("\nAccuracy:", metrics.accuracy_score(y_test, output))
    # print("\nConfusion matrix:\n", metrics.confusion_matrix(y_test, output))

    kf = KFold(n_splits=10,shuffle=True)
    best_accuracy = 0
    best_model = None
    for train_index, test_index in kf.split(matrix_images):
        X_train, X_test = matrix_images[train_index], matrix_images[test_index]
        y_train, y_test = class_label[train_index], class_label[test_index]

        cnnKfold = Model(layers=Layers)
        cnnKfold.fit(features=X_train, target=y_train, batch_size=5, epoch=3, learn_rate=0.1)
        output = np.array([])
        for data in X_test:
            forward_cnn = cnnKfold.forward(data)
            output = np.append(output, np.rint(forward_cnn))
        
        accuracy = metrics.accuracy_score(y_test, output)
        print("\nAccuracy:", accuracy)
        print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
    print("\nBest Accuracy:", best_accuracy)
