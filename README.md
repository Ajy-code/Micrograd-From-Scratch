# micrograd-from-scratch

> **Moteur de différenciation automatique scalaire et étude des systèmes d'optimisation non-linéaires.**

## 📌 Philosophie du projet
Ce projet implémente un moteur de **différenciation automatique en mode inverse** (reverse-mode autodiff) agissant au niveau scalaire. L'objectif est de s'extraire de l'opacité des frameworks modernes (boîtes noires) pour reconstruire les principes fondamentaux de la **rétropropagation du gradient** (backpropagation) au sein de graphes de calcul statiques.

En travaillant exclusivement sur des objets `Value` plutôt que sur des tenseurs, ce moteur permet une analyse granulaire de la **règle de la chaîne** (*chain rule*) et de la sensibilité des paramètres dans une architecture de type Perceptron Multicouche (MLP).



## 🛠️ Spécifications Techniques

### 1. Noyau de calcul (Engine)
* **Graphe Orienté Acyclique (DAG)** : Représentation explicite des opérations arithmétiques. Chaque nœud `Value` stocke sa valeur locale (`data`) et son gradient partiel (`grad`).
* **Tri Topologique** : Implémentation d'un ordonnancement récursif des dépendances garantissant une évaluation exacte des gradients lors de la phase `backward`.
* **Accumulation de Gradient** : Gestion rigoureuse de la confluence des gradients (fan-out > 1) via une accumulation additive, assurant la justesse mathématique du calcul des dérivées partielles lors de l'utilisation multiple d'une même variable.

### 2. Opérateur de Transfert : CReLU (Concatenated ReLU)
Le choix s'est porté sur la **CReLU** comme opérateur de transfert non-linéaire pour ses propriétés structurelles :
* **Mapping avec préservation de phase** : En concaténant $ReLU(x)$ et $ReLU(-x)$, cet opérateur transforme un signal de dimension $n$ en un signal de dimension $2n$. Cela permet de capturer simultanément les composantes positives et négatives du signal.
* **Linéarité par morceaux** : Ce choix facilite le flux de gradient par rapport aux fonctions sigmoïdales saturantes, tout en doublant l'espace des caractéristiques (*feature space*) pour une expressivité accrue.



### 3. Architecture du Modèle (MLP)
* **Structure Feedforward** : Organisation modulaire en couches (`Layer`) et neurones (`Neuron`) où chaque poids et biais est une instance indépendante.
* **Descente de Gradient** : Optimisation itérative par minimisation d'une fonction de perte scalaire (L2 ou Cross-Entropy).
* **Régularisation** : Possibilité d'intégrer une pénalité sur la norme des poids pour contraindre la complexité du modèle.

## 🧪 Exemple d'implémentation

```python
from micrograd.engine import Value

# Définition des entrées
x = [Value(2.0, label='x1'), Value(3.0, label='x2')]

# Construction d'un neurone simple (produit scalaire)
w = [Value(0.1, label='w1'), Value(0.2, label='w2')]
b = Value(0.01, label='bias')
z = sum((wi*xi for wi,xi in zip(w, x)), b)

# Application d'un redressement (base CReLU)
y_pos = z.relu()
y_neg = (-z).relu()

# Rétropropagation automatique
y_pos.backward()

print(f"Gradient du poids w1 : {w[0].grad}")