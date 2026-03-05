# Micrograd from Scratch : Moteur d'Autodifférenciation Scalaire

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fundamentals-success)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)

Ce projet est une implémentation *from scratch* (sans aucune bibliothèque mathématique externe comme NumPy ou PyTorch) d'un moteur d'autodifférenciation scalaire (autograd) et d'un mini-framework de réseaux de neurones (MLP). 

L'objectif de ce projet est de démystifier la mécanique de la rétropropagation (backpropagation) en codant explicitement le graphe de calcul dynamique et la propagation locale des gradients par l'application stricte de la règle de dérivation en chaîne (Chain Rule).

## 🎯 Objectifs techniques atteints

- **Moteur Autograd Scalaire (`micrograd/engine.py`)** : 
  - Création d'une classe `Value` encapsulant une donnée scalaire, son gradient, et l'historique de sa création (parents et opération).
  - Surcharge des opérateurs magiques de Python (`__add__`, `__mul__`, `__pow__`, etc.) pour construire dynamiquement un Graphe Acyclique Dirigé (DAG) lors de la passe avant (*forward pass*).
  - Implémentation du tri topologique pour garantir l'ordre exact de la rétropropagation.
  - Implémentation analytique des dérivées locales pour les opérations de base et les fonctions d'activation non-linéaires (`tanh`, `ReLU`).

- **Mini-Framework Réseau de Neurones (`micrograd/nn.py`)** :
  - Conception modulaire orientée objet : classes `Module`, `Neuron`, `Layer`, et `MLP`.
  - Initialisation aléatoire des poids et des biais en tant qu'objets `Value` traçables.
  - Gestion centralisée des paramètres (`model.parameters()`) et remise à zéro des gradients (`zero_grad()`).

- **Validation et Entraînement (`tests/test_nn.py`)** :
  - Construction d'un réseau multi-couches (ex: `MLP(3, [4, 4, 1])`).
  - Définition d'une fonction de perte (Mean Squared Error - MSE) scalaire.
  - Implémentation d'une boucle d'entraînement complète avec descente de gradient classique (SGD) pour l'optimisation des poids et la validation de l'apprentissage.

## 🧠 Architecture du Graphe de Calcul (L'ingénierie)

Le cœur du projet repose sur la gestion rigoureuse de l'accumulation des gradients. 

Lorsqu'un nœud du graphe est utilisé plusieurs fois (ex: `y = x*x + x`), la dérivée totale par rapport à ce nœud est la **somme** des dérivées partielles provenant de toutes ses branches parentes (règle du calcul multivarié). L'implémentation garantit cette accumulation via l'opérateur `+=` dans les fonctions `_backward` locales, évitant ainsi l'écrasement silencieux des gradients.

## 🛠️ Structure du projet

```text
Micrograd-From-Scratch/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py       # Le moteur d'autograd (Classe Value, opérations, backward)
│   └── nn.py           # Le framework MLP (Neuron, Layer, Multi-Layer Perceptron)
├── tests/
│   ├── __init__.py
│   ├── test_engine.py  # Tests unitaires validant l'exactitude des gradients analytiques
│   └── test_nn.py      # Script d'entraînement et test du MLP (Loss MSE, Descente de gradient)
└── README.md