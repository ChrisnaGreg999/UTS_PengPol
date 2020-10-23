# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:14:13 2020

@author: Gregorius Chrisna Mahendra / 170709244
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

dataset = pd.read_csv('dataset100.csv')
#print(dataset.head)
#print(dataset.columns)

X = dataset[['Tekanan Darah', 'Kadar Oksigen Darah', 'Umur',
       'Jumlah Penyakit Komplikasi']].values #parameter
Y = dataset['Resiko'].values #label
#print(X)
#print(Y)


#-----Normalisasi-----
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
#print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
#print ('Train set:', X_train.shape,  Y_train.shape)
#print ('Test set:', X_test.shape,  Y_test.shape)


Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

k = 9
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
print(neigh)

yhat = neigh.predict(X_test)
print(yhat)

print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, yhat))
