#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Claire Ardern 
# Intro to Machine Learning - Homework 3
# Naive Bayesian Classification and PCA
# 10/25/2022


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
np.random.seed(0)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


# In[3]:


cancer = load_breast_cancer(as_frame = True)


# In[4]:


######################
##### PROBLEM 1 ######
######################

# Fit a Naive Bayes model to the data 
model = GaussianNB()
model.fit(cancer.data, cancer.target)
print(model)

# Make predictions
expected = cancer.target
predicted = model.predict(cancer.data)

# Summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[18]:


######################
##### PROBLEM 2 ######
######################

from sklearn.preprocessing import StandardScaler

# Separate out the features
X = cancer.data

# Separate out the target
Y = cancer.target

# Standardize the features 
X = StandardScaler().fit_transform(X)

K = np.array(range(20))
K_max = 20
accuracy = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)

for k in K:

    pca = PCA(n_components = k)
    principalComponents = pca.fit_transform(X)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)
    
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_val)
    
    accuracy[k] = metrics.accuracy_score(Y_val, Y_pred)
    precision[k] = metrics.precision_score(Y_val, Y_pred)
    recall[k] = metrics.recall_score(Y_val, Y_pred)


# In[19]:


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





# In[20]:


######################
##### PROBLEM 3 ######
######################

# Separate out the features
X = cancer.data

# Separate out the target
Y = cancer.target

# Standardize the features 
X = StandardScaler().fit_transform(X)

K = np.array(range(20))
K_max = 20
accuracy = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)

for k in K:

    pca = PCA(n_components = k)
    principalComponents = pca.fit_transform(X)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)
    
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_val)
    
    accuracy[k] = metrics.accuracy_score(Y_val, Y_pred)
    precision[k] = metrics.precision_score(Y_val, Y_pred)
    recall[k] = metrics.recall_score(Y_val, Y_pred)


# In[21]:


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




