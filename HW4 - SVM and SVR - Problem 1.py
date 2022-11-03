#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Claire Ardern 
# Intro to Machine Learning - Homework 4, Problem 1
# SVM and SVR Classification
# 11/03/2022


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
np.random.seed(0)

from scipy import stats
import seaborn as sns; sns.set()
from sklearn.svm import SVC, SVR


# In[3]:


######################
##### PROBLEM 1 ######
######################

cancer = load_breast_cancer(as_frame = True)

X = cancer.data

Y = cancer.target

X = StandardScaler().fit_transform(X)


# In[4]:


# Perform SVM Classification with linear kernel.

K = np.array(range(20))
K_max = 20
accuracy = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)

for k in K:

    pca = PCA(n_components = k)
    principalComponents = pca.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)
    
    # Linear Support vector classifier
    classifier = SVC(kernel = 'linear', C = 1E10)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    
    accuracy[k] = metrics.accuracy_score(Y_test, Y_pred)
    precision[k] = metrics.precision_score(Y_test, Y_pred)
    recall[k] = metrics.recall_score(Y_test, Y_pred)


# In[5]:


# Plot accuracy, precision, and recall over different number of Ks.

plt.figure()
plt.plot(range(1, K_max+1), accuracy, color = 'red', label = "Accuracy")
plt.plot(range(1, K_max+1), precision, color = 'blue', label = "Precision")
plt.plot(range(1, K_max+1), recall, color = 'green', label = "Recall")
plt.grid()
plt.xlabel('Principal Components')
plt.ylabel('Metrics')
plt.title('Accuracy, Precision, and Recall over Principal Components')
plt.legend()


# In[6]:


#plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap = 'autumn')
#plot_svc_decision_function(classifier)


# In[7]:


# Perform SVM Classification with rbf kernel.

K = np.array(range(20))
K_max = 20
accuracy = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)

for k in K:

    pca = PCA(n_components = k)
    principalComponents = pca.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)
    
    # Linear Support vector classifier
    classifier = SVC(kernel = 'rbf', C = 1E10)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    
    accuracy[k] = metrics.accuracy_score(Y_test, Y_pred)
    precision[k] = metrics.precision_score(Y_test, Y_pred)
    recall[k] = metrics.recall_score(Y_test, Y_pred)


# In[8]:


# Plot accuracy, precision, and recall over different number of Ks.

plt.figure()
plt.plot(range(1, K_max+1), accuracy, color = 'red', label = "Accuracy")
plt.plot(range(1, K_max+1), precision, color = 'blue', label = "Precision")
plt.plot(range(1, K_max+1), recall, color = 'green', label = "Recall")
plt.grid()
plt.xlabel('Principal Components')
plt.ylabel('Metrics')
plt.title('Accuracy, Precision, and Recall over Principal Components')
plt.legend()


# In[9]:


# Perform SVM Classification with poly kernel.

K = np.array(range(20))
K_max = 20
accuracy = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)

for k in K:

    pca = PCA(n_components = k)
    principalComponents = pca.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)
    
    # Linear Support vector classifier
    classifier = SVC(kernel = 'poly', C = 1E10)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    
    accuracy[k] = metrics.accuracy_score(Y_test, Y_pred)
    precision[k] = metrics.precision_score(Y_test, Y_pred)
    recall[k] = metrics.recall_score(Y_test, Y_pred)


# In[10]:


# Plot accuracy, precision, and recall over different number of Ks.

plt.figure()
plt.plot(range(1, K_max+1), accuracy, color = 'red', label = "Accuracy")
plt.plot(range(1, K_max+1), precision, color = 'blue', label = "Precision")
plt.plot(range(1, K_max+1), recall, color = 'green', label = "Recall")
plt.grid()
plt.xlabel('Principal Components')
plt.ylabel('Metrics')
plt.title('Accuracy, Precision, and Recall over Principal Components')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




