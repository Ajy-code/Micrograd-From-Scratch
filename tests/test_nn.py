import sys
import os
# 1. Récupère le chemin absolu du dossier contenant ce script (le dossier 'tests')
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Remonte d'un niveau pour atteindre la racine du projet ('Micrograd-From-Scratch')
parent_dir = os.path.dirname(current_dir)
# 3. Ajoute la racine au path
sys.path.append(parent_dir)
import micrograd.nn as nn

# petit dataset pour faire quelque test
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
#Les valeur réelles (à prédire)
ys = [1.0, -1.0, -1.0, 1.0]
#Création d'un model MLP
model=nn.MLP(3, [4, 4, 1])
ypred=[model(x) for x in xs]
print(ypred)
#loss MSE
def loss(ys,ypred):
    loss=(ypred[0] - ys[0])**2
    for i in range(1,len(ys)):
        loss+=(ypred[i]-ys[i])**2
    return loss
def one_boucle_train(model,xs,ys,lr):
    model.zero_grad()
    ypred=[model(x) for x in xs]
    loss1=loss(ys,ypred)
    loss1.backward()
    for p in model.parameters():
        p.data -= lr*p.grad

#entrainement du modèle (500 boucles)
for i in range(500):
    one_boucle_train(model,xs,ys,0.05)

#Pour comparer avec ypred (celui avec le modèle pas entrainé) et voir si l'entrainement a fonctionné    
ypred2=[model(x) for x in xs]
print(ypred2)
