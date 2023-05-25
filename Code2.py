#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:13:01 2023

@author: nico
"""
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree


raw_data = pd.read_csv("dataset.csv")

#%% Présenation des données
raw_data.head()

#age - age in years
#sex - (1 = male; 0 = female)
#cp - chest pain type
#trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#chol - serum cholestoral in mg/dl
#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#restecg - resting electrocardiographic results
#thalach - maximum heart rate achieved
#exang - exercise induced angina (1 = yes; 0 = no)
#oldpeak - ST depression induced by exercise relative to rest
#slope - the slope of the peak exercise ST segment
#ca - number of major vessels (0-3) colored by flourosopy
#thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
#target - have disease or not (1=yes, 0=no)

raw_data.target.value_counts()


sns.countplot(x="target", data=raw_data)
plt.title("Representation du nombre d'individus malades et sains")
plt.show()

nbsains = len(raw_data[raw_data.target == 0])
nbmalades = len(raw_data[raw_data.target == 1])
print("Pourcentage de patients sains: {:.2f}%".format((nbsains / (len(raw_data.target))*100)))
print("Pourcentage de patients malades: {:.2f}%".format((nbmalades / (len(raw_data.target))*100)))

nbfemme = len(raw_data[raw_data.sex == 0])
nbhomme = len(raw_data[raw_data.sex == 1])
print("Pourcentage de femmes: {:.2f}%".format((nbfemme / (len(raw_data.sex))*100)))
print("Pourcentage d'hommes: {:.2f}%".format((nbhomme / (len(raw_data.sex))*100)))

#%%
plt.close("all")
crosstab=pd.crosstab(raw_data["target"],raw_data["sex"])
print(crosstab)
ax=crosstab.plot(kind='bar',stacked=True,)
for c in ax.containers:
    # set the bar label
    ax.bar_label(c, label_type='center')
# add annotations if desired
plt.title("Histogramme de repartition des sexes en fonction de target")
plt.xlabel("target")
plt.ylabel("Nombres d'individus")
plt.legend(["Homme","Femme"])
plt.show()

#Le dataset ne prend pas en compte beaucoup de femme saines. ("est il representatif ??")

#%% Moyennes
raw_data.groupby('target').mean()

#%%

plt.scatter(x=raw_data.age[raw_data.target==0], y=raw_data.thalach[(raw_data.target==0)],c="green")
plt.scatter(x=raw_data.age[raw_data.target==1], y=raw_data.thalach[(raw_data.target==1)], c="red")
plt.legend(["Sain", "Malade"])
plt.xlabel("Age")
plt.ylabel("BPM max")
plt.title("Representation des pulsations max observées chez les patients en fonction de leur age et de target")
plt.show()

#%% etude de target en fonction du type de douleur
plt.close("all")
pd.crosstab(raw_data.cp,raw_data.target).plot(kind="bar",figsize=(15,6),color=['green','red'])
plt.title(' Maladie cardiaque en fonction du type de douleur')
plt.xlabel('Type de douleur')
plt.xticks(rotation = 0)
plt.legend(["Sain","Malade"])
plt.ylabel("Nombres d'individus")
plt.show()

#%%arbre de decision
#raw_data[["cp","thal","slope"]]=raw_data[["cp","thal","slope"]].astype("category")

y = raw_data.target.values
x_data = raw_data.drop(['target'], axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#%%on evite de faire ca enfaite

for i in range (1,10):
    for j in range (1,5):
        dtc = DecisionTreeClassifier(max_depth=i,min_samples_leaf=j)
        dtc.fit(x_train, y_train)
        #tree.plot_tree(dtc)
        
        acc = dtc.score(x_test, y_test)*100
        #print("Decision Tree depth_max={} min_sample_leaf={} Test Accuracy {:.2f}%".format(i,j,acc))

#%%
dtc = DecisionTreeClassifier(max_depth=None,min_samples_leaf=1,criterion='entropy',ccp_alpha=0.020)
dtc.fit(x_train, y_train)
tree.plot_tree(dtc)
acc = dtc.score(x_test, y_test)*100
print("Decision Tree, Test Accuracy {:.2f}%".format(acc))


