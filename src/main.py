from operator import le
from layer import *
from generateImage import *
from sklearn import metrics

DATA_PATH = "./test"

if __name__ == "__main__":
    def fit(layers, features, target, batch_size, epoch, learn_rate, momentum=1):
        y = np.array()
        output = np.array()

        for e in range(epoch):
            print("Epoch: ", e+1)
            sum_loss = 0
            
            for b in range(batch_size):
                current_idx = (batch_size * e + b) % len(features)
                for layer in layers:
                    res = layer.forward(features[current_idx])
                current_output = res[len(res) - 1][0]
                current_y = target[current_idx]
                
                dE = np.array([current_y - current_output]) * -1
                for i in reversed(range(len(layers))):
                    dE = layers[i].backward(dE)
                sum_loss += 0.5 * (current_y - current_output)**2
                
                y = np.append(y, current_y)
                output = np.rint(np.append(output, current_output))

            for l in reversed(range(len(layers))):
                layers[l].update_weights(learn_rate, momentum)
            avg_loss = sum_loss/batch_size
            
            print("Loss: ", avg_loss)
            print("Accuracy: ", metrics.accuracy_score(y, output))
                
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