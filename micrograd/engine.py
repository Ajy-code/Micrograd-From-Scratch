"""Création de la class Value, elle permet de retracer "l'histoire" des 
valeurs scalaire, cela va permettre de faire la backpropagation"""
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
