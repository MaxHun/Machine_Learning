import numpy as np

class NeuralNet(object):
    
    def __init__(self, len_input=1, layers=None, loadfile = None, savefile=None, loss_type="binary_cross_entropy", alpha=0.1): # NeuralNet([[128, "logistic"], ...], )
        self.alpha = alpha
        self.layers = []
        if savefile != None:
            self.savefile = savefile
            
        if loadfile == None: # initialise everything with weight 0
            dim_in = len_input
            for layer in layers:
                self.add_layer(layer, dim_in)
                dim_in = layer[0]
            self.dim_out = dim_in
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
                    self.dim_out = num_neurons
                    break
        self.dim_in = self.layers[0].num_neurons
        self.inputs = np.zeros(self.dim_in)
        self.loss_type = loss_type
        self.loss_fct = self.get_loss_fct()
        self.num_layers = len(self.layers)
        
    def get_loss_fct(self):
        def binary(x, y):
            return -y*np.log(x) - (1 - y)*np.log(1-x)
        which_function = {"binary_cross_entropy" : binary}
        
        return which_function[self.loss_type]
        
    def delta_L(self, y_hat, y): 
        if self.loss_type == "binary_cross_entropy": # only works with sigmoid activation
            return 1/len(y_hat)*(y_hat - y)
        
        
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
            curr = self.layers[index].forward(curr)
        return curr
        
    def run_badge(self, badge):
        
        y = np.zeros((self.dim_out, len(badge)))
        for i, x in enumerate(badge):
            self.inputs += x
            y[i,:] = self.run_instance(x)
        return y
        
    def delete_inputs(self):
        self.inputs = np.zeros(len(self.inputs))
        
    def delete_activations(self):
        for layer in self.layers:
            layer.delete_activations()
        self.delete_inputs()
        

    def train_on_batch(self, x_batch, y_batch):
        max_der = 100
        while max_der > 1:
            self.delete_activations()
            y_hat = self.run_badge(x_badge)
            print(f"The cost is {sum(self.loss_fct(y_hat, y_batch))\
                                /len(self.loss_fct(y_hat, y_batch))}")
            dc_dw, dc_dw = self.backward(y_hat, y_batch)
            for layer, i in enumerate(self.num_layers):
                layer.weights -= self.alpha/len(x_batch)*dc_dw[i]
                layer.bias -= self.alpha/len(x_batch)*dc_db[i]
            max_der = max([deri.max for deri in dc_dw+dc_db])
            
        
    def backward(self, x, y):
        max_der = 100
        dc_db = [np.zeros(len(layer.bias)) for layer in self.layers]
        dc_dw = [np.zeros(layer.weights.shape) for layer in self.layers]
        dc_db[-1] = delta_L(y_hat, y)
        for i in range(len(delta)-1, -1, -1):
            dc_db[i] = np.matmul(self.layers[i+1].weights.T, delta[i+1])\
                       *self.layers[i].act_fct_der(self.layers[i].zs)
            if i > 0:
                dc_dw[i] = np.matmul(self.layers[i-1].activations, dc_db[i]) 
            else:
                dc_dw[i] = np.matmul(self.inputs, dc_db[i])
        return dc_dw, dc_db
        
        
    def train(self, x, y, badge_size=1, num_epochs=1):
        rng = np.random.default_rng(1)
        for e in range(num_epochs):
            print(f"Starting epoch {e}...")
            p = rng.permutation(len(x))
            x = x[p]
            y = y[p]
            for x_batch, y_batch in zip(np.array_split(x, batch_size),\
                                        np.array_split(y, batch_size)):
                self.train_on_batch(x_batch, y_batch)
    
    def convert_result(self, y, conversion="round"):
        if conversion == "round":
            return np.roound(y)
                
    def test(self, x_test, y_test):
        errors = dict()
        for x, y in zip(x_text, y_test):
            y_hat = self.run_instance(x)
            if not y == y_hat:
                errors[(y,y_hat)] = errors[(y,y_hat)] + 1 if (y,y_hat)\ß
                                    in errors else 1
        num_errors = sum([errors[i] for i in errors])
        print(f"We tested {len(x_test)} many instances, out of which" +\
              f"{num_errors} were erroneous, which corresponds to an " +\
              f"accuracy of {num_errors/len(x_test)}")
        for e, o in errors:
            print(f"{errors[(e,o)]} many instances gave {o}, while " +\
                  f"{e} was expected.")
        
        
        
    
        

        
        
        
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
        self.num_neurons = num_neurons
        self.act_fct = self.functions(neu_type)
        self.act_fct_der = self.functions(neu_type, der=True)
        self.activations = np.zeros(num_neurons)
        self.zs = np.zeros(num_neurons)
        #self.batch_size = 0
        self.neu_type = neu_type
    
    def forward(self, x):
        z = np.matmul(self.weights, x) + b
        self.zs += z
        a = self.act_fct(z)
        self.activations += a
        #self.batch_size += 1
        return a
    
    def delete_activations(self):
        self.activations = np.zeros(len(self.activations))
        #self.batch_size = 0
        self.zs = np.zeros(len(self.zs))
    
    def functions(self, act_fct_str, der=False):
        
        act_fct_str += "_der"*der
        
        def sigmoid(x):
            return 1/(1+np.exp(x))
        
        def relu(x):
            return max(0,x)
            
        def sigmoid_der(x):
            return sigmoid(x)*(1 - sigmoid(x))
            
        
        which_function = {"logistic" : sigmoid, "relu" : relu, \
                         "logistic_der" : sigmoid_der}
        
        return which_function[act_fct_str]
        
        
        
        
if __name__ == "__main__":
    N = NeuralNet(1, [[4, "logistic"],[4, "logistic"], [1, "logistic"]],\
                  loadfile=None, savefile="trained/test")
    
    
    
    N.train(x_train, y_train, badge_size = 100, num_epochs = 6)
    N.test(x_text, y_test)
