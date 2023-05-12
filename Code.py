#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:14:05 2023

@author: nico
"""

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt  
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X, y = iris.data, iris.target


#%%

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y, sample_weight=None, check_input=True)
tree.plot_tree(clf)
plt.title('Arbre de décision par defaut sur le jeu de donnée IRIS')
plt.show()


#%%

clf = DecisionTreeClassifier(random_state=0,max_depth=1,min_samples_leaf=1)
#max_depth -> The maximum depth of the tree (none = pure leaf at the end of every branch)
#min_sample_leaf -> The minimum number of samples required to be at a leaf node

clf.fit(X, y, sample_weight=None, check_input=True)
tree.plot_tree(clf)
plt.title('Arbre de décision avec profondeur maximale=1 sur le jeu de donnée IRIS')
plt.show()



#%%

clf = DecisionTreeClassifier(random_state=0,max_depth=None,min_samples_leaf=20)
#max_depth -> The maximum depth of the tree (none = pure leaf at the end of every branch)
#min_sample_leaf -> The minimum number of samples required to be at a leaf node

clf.fit(X, y, sample_weight=None, check_input=True)
tree.plot_tree(clf)
plt.title('Arbre de décision avec elements par feuille>=20 sur le jeu de donnée IRIS')
plt.show()

#%%
