import math
from treelib import Tree
from typing import List, Union


class Value:
    """
    Values are organised and connected in a graph (more specifically, a tree). A value is nothing more than a float, 
    that can be combined with other values or floats via basic mathematical operations. 
    """
    def __init__(self, data: float, children: tuple['Value'] = (), operator: str = "", label: str ="") -> None:
        self.data: float = data
        self.grad: float = 0.0
        # Backpropagation function that calculates gradient of each Value relative to the root of the tree
        # - uses the Chain rule from Calc: dz/dx = dz/dy * dy/dx (https://en.wikipedia.org/wiki/Chain_rule)
        # - allows us to see each Values final impact on the root of the tree (final nodes)
        # - numeric Values (such as biases) don't have a backprop function, thus we init with None
        # - gradients will be summed, as any node x(i)(k), where i is any layer of our neural net and 
        #   k is any index of a node in layer i, depends mathematically on all previous nodes in layer i-1 
        self._backprop = lambda: None
        self._prev  = set(children)
        self.operator = operator
        self.label = label

    """
    Mathematical operations
    """
    def __neg__(self) -> 'Value':
        return self * -1

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backprop():
            # given a sum: x + y = z; we see that dz/dx = 1 && dz/dy = 1
            self.grad  += out.grad
            other.grad += out.grad

        out._backprop = _backprop
        return out
    
    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)
    
    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backprop():
            # given a product: x * y = z; we see that dz/dx = y && dz/dy = x
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        
        out._backprop = _backprop
        return out

    def __rmul__(self, other: 'Value') -> 'Value':
        # if we call int/float * Value, this function will be a fallback and __mul__ will be called
        return self * other
    
    def __truediv__(self, other: 'Value') -> 'Value':
        return self * other**(-1)
    
    def __pow__(self, other: Union[int, float]) -> 'Value':
        assert isinstance(other, (int, float), "currently only supporting int/float powers")
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backprop():
            # given x**c = x^c = z; we see that dz/dx = c * x**(c-1) = c * x^(c-1)
            self.grad = other * self.data**(other-1) * out.grad

        out._backprop = _backprop
        return out
    
    def exp(self) -> 'Value':
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backprop():
            # given e**x = e^x = z; we see that dz/dx = e**x = e^x
            self.grad += out.data * out.grad
        
        out._backprop = _backprop
        return out

    """
    Backpropagation
    """
    def backprop(self) -> None:
        # implements a topological sort instead of using a recusive function
        sorted_nodes = []
        visited_nodes = set()

        def build_topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._prev:
                    build_topo(child)
                sorted_nodes.append(node)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backprop()

    def backprop_recursive(self) -> None:
        # implements a recurisve backpropagation
        if self.operator == "tanh":
            self.grad = 1.0
        else:
            self._backprop()

        for child in self._prev:
            child.backprop_recursive()

    """
    Activation functions
    """
    def tanh(self) -> 'Value':
        x = self.data
        # tanh x  (https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions)
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backprop():
            # see https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives
            self.grad = (1 - t**2) * out.grad

        out._backprop = _backprop
        return out
    
    """
    Representation
    """
    def __repr__(self) -> str:
        return f"Value({f'label={self.label}, ' if self.label else ""}data={self.data}, grad={self.grad}{f', operator="{self.operator}"' if self.operator else ""})"

    def cli_print_tree(self, tree=None, parent=None) -> None:
        is_new_tree = False

        if not tree:
            is_new_tree = True
            tree = Tree()
            parent = tree.create_node(tag=self.__repr__())

        for child in self._prev:
            curr = tree.create_node(tag=child.__repr__(), parent=parent)
            child.print_tree_recursive(tree, curr)
        
        if is_new_tree:
            print(tree.show(stdout=False))