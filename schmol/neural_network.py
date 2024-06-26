import random

from schmol.expressions import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        []


class Trainable:
    def train(data, labels):
        pass


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


class MLP(Module, Trainable):
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

