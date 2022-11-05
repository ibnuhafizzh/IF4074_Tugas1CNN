import numpy as np
import common
from common import sigmoid, relu


class DenseLayer:
    def __init__(self, n_unit, activation):
        self.n_unit = n_unit
        self.activation = activation
        self.bias = np.zeros(n_unit)
        self.weight = []
        self.deltaW = np.zeros(n_unit)

    def init_weights(self, n_inputs):
        self.n_inputs = n_inputs
        self.weight = np.random.randn(self.n_unit,n_inputs)
        self.deltaW = np.zeros((self.n_unit))

    def forward(self,inputs):
        if len(self.weight) == 0:
           self.init_weights(len(inputs))
        self.input = inputs
        multisum = np.array([])
        for i in range(self.n_unit):
            multisum = np.append(multisum, 
            np.sum(np.multiply(self.weight[i], inputs)) + self.bias[i])

        if self.activation == 'sigmoid':
            matrixsigmoid = np.vectorize(common.sigmoid)
            self.output = matrixsigmoid(multisum)
        else:
            self.output = np.maximum(multisum, 0)
        return self.output
    
    def d_sigmoid(self, inputs):
        sigm = common.sigmoid(inputs)
        return sigm * (1 - sigm)
    
    def d_relu(self,input):
        return 1.0 if input >= 0 else 0

    def d_act_funct(self,activation,inputs):
        if (activation=='sigmoid'):
            return self.d_sigmoid(inputs)
        else:
            return self.d_relu(inputs)

    def backward(self,inputs):
        derivative = np.array([])
        for i in self.output:
            derivative = np.append(derivative, self.d_act_funct(self.activation, i))
        self.deltaW += np.multiply(derivative, inputs)
        dE = np.matmul(inputs, self.weight)
        return dE
    
    def update_weights(self, learn_rate, momentum):
        for i in range(self.n_unit):
            self.weight[i] = self.weight[i] - ((momentum * self.weight[i]) + (learn_rate * self.deltaW[i] * self.input))

        self.bias = self.bias - ((momentum * self.bias) + (learn_rate * self.deltaW))
        self.deltaW = np.zeros((self.n_unit))

class ConvolutionLayer:
    def __init__(self, nb_channel, nb_filter, filter_size, padding=0, stride=1):
        self.nb_channel = nb_channel
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.weight = np.random.randn(nb_filter, nb_channel, filter_size, filter_size)
        self.bias = np.zeros((nb_filter))
        self.dw = np.zeros((nb_filter, nb_channel, filter_size, filter_size))
        self.db = np.zeros((nb_filter))
    
    def add_zero_padding(self, inputs):
        w_pad = inputs.shape[1] + self.padding * 2
        h_pad = inputs.shape[2] + self.padding * 2
        
        inputs_padded = np.zeros((inputs.shape[0], w_pad, h_pad))
        for s in range(inputs.shape[0]):
            inputs_padded[s, self.padding:w_pad-self.padding, self.padding:h_pad-self.padding] = inputs[s, :, :]

        return inputs_padded
    
    def update_weights(self, learn_rate, momentum):
        self.weight -= learn_rate * self.dw
        self.bias -= learn_rate * self.db

        # Reset error
        self.dw = np.zeros((self.nb_filter, self.nb_channel, self.filter_size, self.filter_size))
        self.db = np.zeros((self.nb_filter))
    
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

    def backward(self, prev_errors):
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weight.shape)
        db = np.zeros(self.bias.shape)

        f, w, h = prev_errors.shape

        for k in range(f):
            db[k] = np.sum(prev_errors[k, :, :])
        
        for k in range(f):
            for i in range(w):
                for j in range(h):
                    dw[k, :, :, :] += prev_errors[k, i, j] * self.inputs[:, i:i+self.filter_size, j:j+self.filter_size]
                    dx[:, i:i+self.filter_size, j:j+self.filter_size] += prev_errors[k, i, j] * self.weight[k, :, :, :]

        self.dw += dw
        self.db += db
        
        return self.detector(dx)

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
            for j in range(0, W, self.filter_size):
                for k in range(0, H, self.filter_size):
                    st = np.argmax(self.input[i, j : j + self.filter_size, k : k + self.filter_size])
                    (idx, idy) = np.unravel_index(st, (self.filter_size, self.filter_size))
                    if ((j + idx) < W and (k + idy) < H):
                        dx[i, j + idx, k + idy] = prev_errors[i, int(j / self.filter_size) % prev_errors.shape[1], int(k / self.filter_size) % prev_errors.shape[2]]
        return dx
    
    def update_weights(self, learn_rate, momentum):
        # Todo
        pass

class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        self.channel, self.width, self.height = inputs.shape
        output = inputs.flatten()
        return output
    
    def backward(self, inputs):
        return inputs.reshape(self.channel, self.width, self.height)

    def update_weights(self, learning_rate, momentum):
        pass

class LSTMLayer:
    def __init__(self, input_size, n_cells):
        self.input_size = input_size
        self.n_cells = n_cells

        self.c_prev = np.zeros(n_cells,1)
        self.h_prev = np.zeros(n_cells,1)

        self.sigmoid = np.vectorize(common.sigmoid) # allow function to receive input in form of vector

        #parameternya dijadiin class
        self.input_param = self.LSTMParameter(self.input_size, self.n_cells)
        self.cell_param = self.LSTMParameter(self.input_size, self.n_cells)
        self.forget_param = self.LSTMParameter(self.input_size, self.n_cells)
        self.output_param = self.LSTMParameter(self.input_size, self.n_cells)
        self.training_param = {}
    
    class LSTMParameter:
        def __init__(self,size,n_cells):
            self.u = np.random.rand(n_cells, size)
            self.w = np.random.rand(n_cells)
            self.b = np.random.rand(n_cells)


    def forgetGate(self, timestep):
        self.training_param['f'+str(timestep)] = self.sigmoid(
                np.dot(self.forget_param.u, self.x[timestep]) + 
                np.dot(self.forget_param.w, self.h_prev) + 
                self.forget_param.b
            )

    def inputGate(self,timestep):
        self.training_param['i' + str(timestep)] = self.sigmoid(
            np.matmul(self.input_param.u,self.x[timestep]) 
            + np.multiply(self.input_param.w,self.h_prev[timestep]) 
            + self.input_param.b)
    
    def cellState(self,timestep):
        self.parameter['Caccent'+str(timestep)] = np.tanh(
            np.dot(self.cell_param.u, self.x[timestep]) + np.dot(self.cell_param.w, self.h_prev) + self.cell_param.b)
        self.parameter['C'+str(timestep)] = (np.multiply(
            self.parameter['f'+str(timestep)], self.c_prev) + 
            np.multiply(self.parameter['i'+str(timestep)], self.parameter['Caccent'+str(timestep)]))

    def outputGate(self, timestep):
        self.training_param['o'+str(timestep)] = self.sigmoid(
                np.dot(self.output_param.u, self.x[timestep]) + 
                np.dot(self.output_param.w, self.h_prev) + 
                self.output_param.b
            )
        self.training_param['h'+str(timestep)] = np.multiply(self.training_param['o'+str(timestep)], np.tanh(self.training_param['C'+str(timestep)]))
    
    def forward(self, inputs):
        self.x = inputs
        for i in range(self.n_cells):
            self.forgetGate(i)
            self.inputGate(i)
            self.cellState(i)
            self.outputGate(i)

            self.c_prev = self.training_param['C'+str(i)]
            self.h_prev = self.training_param['h'+str(i)]
        
        output = self.training_param['h'+str(self.n_cells-1)]
        return output

    def backward(self, inputs):
        # engga ada di spek
        pass

    def update_weights(self, learning_rate, momentum):
        # engga ada di spek
        pass


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
    