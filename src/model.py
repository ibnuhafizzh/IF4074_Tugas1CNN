import pickle
from layer import *
from sklearn import metrics

class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def save_model(self, namefile):
        with open(namefile, 'wb') as f:
            pickle.dump(self.layers,  f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self,namefile):
        with open(namefile, 'rb') as f:
            temp = pickle.load(self.layers,  f, protocol=pickle.HIGHEST_PROTOCOL)
        self.self.layers = temp.copy()
    
    def fit(self,features, target, batch_size, epoch, learn_rate, momentum=1):
        y = np.array([])
        output = np.array([])

        for e in range(epoch):
            print("Epoch: ", e+1)
            sum_loss = 0
            
            for b in range(batch_size):
                current_idx = (batch_size * e + b) % len(features)
                for layer in self.layers:
                    res = layer.forward(features[current_idx])
                current_output = res[len(res) - 1]
                current_y = target[current_idx]
                
                dE = np.array([current_y - current_output]) * -1
                for i in reversed(range(len(self.layers))):
                    dE = self.layers[i].backward(dE)
                sum_loss += 0.5 * (current_y - current_output)**2
                
                y = np.append(y, current_y)
                output = np.rint(np.append(output, current_output))

            for l in reversed(range(len(self.layers))):
                self.layers[l].update_weights(learn_rate, momentum)
            avg_loss = sum_loss/batch_size
            
            print("Loss: ", avg_loss)
            print("Accuracy: ", metrics.accuracy_score(y, output))