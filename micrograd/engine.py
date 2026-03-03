"""Création de la class Value, elle permet de retracer "l'histoire" des 
valeurs scalaire, cela va permettre de faire la backpropagation"""
import math
class Value: 
    def __init__(self, data, _parents=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label
    # Fonction permettant d'afficher, la valeur de self et le gradient de la loss par rapport à self
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    # Fonction permettant de faire des additions
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        #Impact de l'addition sur les gradients
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    #Fonction permettant de faire des multiplications
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        #Impact de la multiplication sur les gradients
        def _backward():
            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data
        out._backward=_backward
        return out
    #Utilisation de la commutativité de l'addition
    def __radd__(self, other):
        return self + other
    #Utilisation de la commutativité de la multiplication
    def __rmul__(self, other):
        return self*other
    #Mise en place des fonction permettant les opérations usuelles restantes
    def __neg__(self):
        out=Value(-self.data,(self,),'-')
        def _backward():
            self.grad += -out.grad
        out._backward=_backward
        return out
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    #Faire attention, la soustraction n'est pas commutative, je ne peux pas faire comme pour l'addition et la multiplication
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out=Value(self.data/other.data,(self,other),'/')
        def _backward():
            self.grad += out.grad/other.data
            other.grad += -self.data*out.grad/(other.data**2)
        out._backward=_backward
        return out
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other/self
    def __pow__(self, other):
        # Je considère que other est un scalaire réel
        assert isinstance(other, (int, float))
        out=Value(self.data**other,(self,),'**')
        def _backward():
            self.grad += other*out.grad*(self.data**(other-1))
        out._backward=_backward
        return out
    #Mise en place de fonction non-linéaire
    def tanh(self):
        out=Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1),(self,),'tanh')
        def _backward():
            self.grad += out.grad*(1-out.data**2)
        out._backward=_backward
        return out
    def relu(self):
        out=Value(max(0,self.data),(self,),'relu')
        def _backward():
            if self.data>0:
                self.grad += out.grad
        out._backward=_backward
        return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parents in v._prev:
                    build_topo(parents)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for i in range(len(topo)):
            topo[len(topo)-i-1]._backward()

            

