import sys
import os
sys.path.append(os.path.abspath("../")) # "../" remonte d'un dossierr

from micrograd.engine import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a * b + c
e = d.tanh()

e.backward()

print(a, b, c, d, e)