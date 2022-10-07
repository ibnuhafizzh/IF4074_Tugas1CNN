import numpy as np
import common


class DenseLayer:
    def __init__(self, n_unit, activation):
        self.n_unit = n_unit
        self.activation = activation
        self.bias = np.zeros(n_unit)
        self.weight = np.random.randn(n_unit)

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
    def __init__(self, nb_channel, nb_filter, filter_size, padding=0, stride=1):
        self.nb_channel = nb_channel
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.weight = np.random.randn(nb_filter, nb_channel, filter_size, filter_size)
        self.bias = np.zeros((nb_filter))
    
    def add_zero_padding(self, inputs):
        w_pad = inputs.shape[1] + self.padding * 2
        h_pad = inputs.shape[2] + self.padding * 2
        
        inputs_padded = np.zeros((inputs.shape[0], w_pad, h_pad))
        for s in range(inputs.shape[0]):
            inputs_padded[s, self.padding:w_pad-self.padding, self.padding:h_pad-self.padding] = inputs[s, :, :]

        return inputs_padded
    
    def forward(self, inputs):
        w, h = inputs.shape[1], inputs.shape[2]
        v_w = int((w - self.filter_size + 2*self.padding)/self.stride + 1)
        v_h = int((h - self.filter_size + 2*self.padding)/self.stride + 1)
        
        self.inputs = self.add_zero_padding(inputs)
        featureMap = np.zeros((self.nb_filter, v_w, v_h))

        for k in range(self.nb_filter):
            for i in range(v_w):
                for j in range(v_h):
                    recField = self.inputs[:, i:i+self.filter_size, j:j+self.filter_size]
                    featureMap[k, i, j] = np.sum(recField * self.weight[k, :, :, :] + self.bias[k])
        
        return self.detector(featureMap)

    def detector(self,input):
        return np.maximum(input, 0)

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

    def backward(self, prev_errors):
        F, W, H = self.input.shape
        dx = np.zeros(self.input.shape)
        for i in range(0, F):
            for j in range(0, W, self._filter_size):
                for k in range(0, H, self._filter_size):
                    st = np.argmax(self.input[i, j : j + self._filter_size, k : k + self._filter_size])
                    (idx, idy) = np.unravel_index(st, (self._filter_size, self._filter_size))
                    if ((j + idx) < W and (k + idy) < H):
                        dx[i, j + idx, k + idy] = prev_errors[i, int(j / self._filter_size) % prev_errors.shape[1], int(k / self._filter_size) % prev_errors.shape[2]]
        return dx
    
    def update_weights(self, learning_rate):
        # Todo
        pass

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

    matrix = np.array([[[1,7,2],[11,1,23],[2,2,2]],[[1,5,2],[10,1,20],[4,2,4]],[[6,7,8],[12,4,6],[8,2,6]]])
    conv = ConvolutionLayer(3, 2, 2, 1)
    print(conv.forward(matrix))
    