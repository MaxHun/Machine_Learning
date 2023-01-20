import numpy as np

class NeuralNet(object):
    
    def __init__(self, len_input=None, layers=None, loadfile = None, savefile=None, loss_type="binary_cross_entropy"): # NeuralNet([[128, "logistic"], ...], )
        self.layers = []
        if savefile != None:
            self.savefile = savefile
            
        if loadfile == None: # initialise everything with weight 0
            dim_in = len_input
            for layer in layers:
                self.add_layer(layer, dim_in)
                dim_in = layer[0]
        else:  # read whole net from file, including weights
            data = np.load(loadfile)
            i = 0
            while True:
                try:
                    weights = data[f"W_{i}"]
                    bias = data[f"b_{i}"]
                    layer_type = data[f"layer_type{i}"]
                    dim_in, num_neurons = weights.shape
                    self.add_layer([num_neurons, layer_type], dim_in)
                    self.layers[-1].weights = weights
                    self.layers[-1].bias = bias
                    i += 1
                except KeyError:
                    break
        self.loss_fct = self.loss_functions(loss_type)
        self.num_layers = len(self.layers)
        
    def loss_functions(self, loss_str):
        def binary(x, y):
            return -y*np.log(x) - (1 - y)*np.log(1-x)
        which_function = {"binary_cross_entropy" : binary}
        
        return which_function[loss_str]
        
    def loss_derivatives(self, loss_str): #TODO
        def binary(x, y):
         return 
        
        
    def save_net(self):
        data_weights = {f"W_{i}" : self.layers[i].weights \
                        for i in range(self.num_layers)}
        data_bias = {f"b_{i}" : self.layers[i].bias for i in range(self.num_layers)}
        data_type = {f"layer_type{i}" : self.layers[i].neu_type for i in range(self.num_layers)}
        np.savez_compressed(self.savefile, **data_weights, **data_bias,
                            **data_type)
        
    
    def add_layer(self, layer, dim_in, index=None): # always first add with weight 0
        num_neurons, layer_type = layer
        if index == None:
            index = len(self.layers)
            
        self.layers.insert(index, NeuralLayer(num_neurons, dim_in))
        
    def run_instance(self, x):
        curr = x
        
        for index in range(self.num_layers):
            curr = self.layers.index.forward(curr)
        return curr
        
    def run_badge(self, badge):
        y = np.zeros(len(badge))
        for i, x in enumerate(badge):
            y[i] = self.run_instance(x)
        return y
        

        
        
        
    def backward(self):
        pass
        
    
        

        
        
        
class NeuralLayer(object):
    def __init__(self, num_neurons, dim_in, neu_type="logistic",
                 weights = None, bias=None):
        if weights == None:
            self.weights = np.zeros((dim_in, num_neurons))
        elif weights.shape == (dim_in, num_neurons):
            self.weights = weights
        else:
            raise ValueError("dimension of weights" +\
                             " and input do not match")
        if bias == None:
            self.bias = np.zeros(dim_in)
        elif len(bias) == dim_in:
            self.bias = bias
        else:
            raise ValueError("dimension of bias" +\
                             " and input do not match")
        self.act_fct = self.functions(neu_type)
        self.neu_type = neu_type
    
    def forward(self, x):
        z = np.matmul(self.weights, x) + b
        return self.act_fct(z)
    
    def functions(self, act_fct_str):
        
        def sigmoid(x):
            return 1/(1+np.exp(x))
        
        def relu(x):
            return max(0,x)
            
        
        which_function = {"logistic" : sigmoid, "relu" : relu}
        
        return which_function[act_fct_str]
        
        
        
        
if __name__ == "__main__":
    NeuralNet(3, [[4, "logistic"],[4, "logistic"]], loadfile=None, savefile="trained/test").save_net()
    NeuralNet(loadfile="trained/test.npz")
