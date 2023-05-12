#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:14:05 2023

@author: nico
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y, sample_weight=None, check_input=True)


tree.plot_tree(clf)
plt.show()
