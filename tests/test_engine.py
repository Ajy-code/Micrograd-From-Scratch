import sys
import os
# 1. Récupère le chemin absolu du dossier contenant ce script (le dossier 'tests')
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Remonte d'un niveau pour atteindre la racine du projet ('Micrograd-From-Scratch')
parent_dir = os.path.dirname(current_dir)
# 3. Ajoute la racine au path
sys.path.append(parent_dir)

from micrograd.engine import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a * b + c
e = d.tanh()

e.backward()
x = Value(3.0, label='x')
y = x * x + x
y.backward()

print(x.grad)