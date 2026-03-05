import sys
import os
sys.path.append(os.path.abspath("../")) # "../" remonte d'un dossierr
from micrograd.engine import Value
import random

class Module:
    #A chaque itération, il faut réinitialiser les gradients 
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0.0
    def parameters(self):
        return []
    
class Neuron(Module):
    #Le neurones a "nin" entrées (la taille du vecteur ligne d'entrée)
    def __init__(self,nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))
    # Les paramètres d'un neurones sont ces poids et sont biais / rq:le plus permet de concaténer les 2 listes 
    def parameters(self):
        return self.w + [self.b]
    #Permet de faire en sorte que l'appel d'un neurones donne directement son résultat (la fonction d'activation non linéaire est tanh)
    def __call__(self,x):
        n=len(self.w)
        z=self.b
        for i in range(n):
            z += self.w[i]*x[i]
        return z.tanh()

class Layer(Module):
    #Une couche est composé de nout neurones
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]
    #Chaque poid et biais de chaque neurones sont des paramètres (rq: les paramètres sont des Values)
    def parameters(self):
        params=[]
        for n in self.neurons:
            params.extend(n.parameters())
        return params
    def __call__(self,x):
        Z=[n(x) for n in self.neurons]
        if len(self.neurons)==1:
            return Z[0]
        else : return Z

class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers=[]
        nin1=nin
        for k in nouts:
            self.layers.append(Layer(nin1,k))
            nin1=k
    def parameters(self):
        params=[]
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    def __call__(self,x):
        z=x
        for layer in self.layers:
            z=layer(z)
        return z
        


        