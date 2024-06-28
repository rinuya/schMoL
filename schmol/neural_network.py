import random

from schmol.expressions import Value
from schmol.utils import vector_loss


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        []


class Model(Module):

    def __call__(self, inputs):
        pass

    def train(self, data=[], labels=[], iterations=50, batch_size=None, print_loss=False):
        assert len(data) != 0,           "No training data given"
        assert len(data) == len(labels), "Length of data and labels is not the same"
        # split data and it's labels into batches
        z = list(zip(data, labels))
        batches = [z[i : i+batch_size] for i in range(0, len(data), batch_size)] if batch_size else [z]
        # iterate
        print(f"Training Model on {len(data)} datapoints for {iterations} iterations:")
        for k in range(iterations):
            for batch in batches:
                # get batch predictions
                batch_labels = [b_label for (_, b_label) in batch]
                predictions = self.predict([b_data for (b_data, _) in batch])
                # calculate loss: SUM of (predicted_value - actual_value)²
                loss = sum((pred - actual)**2 for (actual, pred) in zip(batch_labels, predictions))
                # backpropagate
                loss.backprop()
                # tweak Model params
                for p in self.parameters():
                    p.data += -0.01 * p.grad
                # zero grad
                self.zero_grad()

            if print_loss: 
                print(k, loss.data)
            
        print(f"\Finished model training.")

    def predict(self, data):
        return [self(x) for x in data]
    
    def predict_one(self, one):
        return self(one)


class Neuron(Module):
    """
    Represents a single Neuron in a Neural Network. Has a bias, 
    """
    def __init__(self, inputs_to_neuron: int):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(inputs_to_neuron)]
        self.bias    = Value(random.uniform(-1, 1))
        self.dim     = inputs_to_neuron

    def __call__(self, inputs):
        # used on instance of Neuron with rounded brackets '()'
        assert(len(inputs) == self.dim), f"""Values passed to Neuron (either from previous layer or from the start) 
            have dimension {len(inputs)}, while the current Neuron only has dimension {self.dim} (num of weights). The 
            amount of values you feed into the neuron need to have the same dimension as it's weights."""
        # w * x + b (dot product + bias)
        activation_value = sum((w_i * x_i for w_i, x_i in zip(self.weights, inputs)), self.bias)
        out = activation_value.tanh()
        return out
    
    def parameters(self):
        # all parameters that we can tweak in a neuron to improve our Neural Network performance
        return self.weights + [self.bias]


class Layer(Module):
    """
    Each layer in a Neural Network is comprised of multiple Neurons. For each layer, 
    we can specify how many inputs the Neuron receives feedback from (weights), and 
    how many Neurons the layer has in total.
    """
    def __init__(self, inputs_per_neuron, number_of_neurons):
        self.neurons = [Neuron(inputs_per_neuron) for _ in range(number_of_neurons)]

    def __call__(self, inputs):
        # put each input value into each neuron
        outs = [n(inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # returns all parameters of each Neuron in this layer
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Model):
    """
    A MLP is a Multilayer Perceptron https://en.wikipedia.org/wiki/Multilayer_perceptron
    """
    def __init__(self, number_of_inputs, hidden_layer_sizes, number_of_outputs) -> None:
        mlp_structure = [number_of_inputs] + hidden_layer_sizes + [number_of_outputs]
        self.inputs        = number_of_inputs
        self.hidden_layers = hidden_layer_sizes
        self.outputs       = number_of_outputs
        self.layers        = [Layer(mlp_structure[i], mlp_structure[i+1]) for i in range(len(hidden_layer_sizes) + 1)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def parameters(self):
        # returns all parameters of each Layer in this MLP
        return [p for layer in self.layers for p in layer.parameters()]
    
    def train(self, data=[], labels=[], iterations=50, batch_size=None, print_loss=False):
        assert len(data[0]) == self.inputs, f"Training data must be a feature vector of size {self.inputs}, but provided data has size {len(data[0])}"
        super().train(data, labels, iterations, batch_size, print_loss) 

    def __repr__(self):
        return f"A MLP with input layer of size {self.inputs}, hidden layers {self.hidden_layers}, and output layer of size {self.outputs}."
