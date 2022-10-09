#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Claire Ardern 
# Intro to Machine Learning - Homework 2
# Logistic Regression and K-Fold Cross-Validation
# 10/06/2022


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets 
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
np.random.seed(0)


# In[3]:


######################
##### PROBLEM 1 ######
######################

# Read in Diabetes data file.
diabetes = pd.read_csv('./diabetes.csv')

X = diabetes.iloc[:, [0,7]].values
Y = diabetes.iloc[:, 8].values

# Create training and validation sets
X_train, X_val, Y_train, Y_val  = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the training data and transform 
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[4]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_val)

cnf_matrix = confusion_matrix(Y_val, Y_pred)

print("Accuracy:", metrics.accuracy_score(Y_val, Y_pred))
print("Precision:", metrics.precision_score(Y_val, Y_pred))
print("Recall:", metrics.recall_score(Y_val, Y_pred))


# In[5]:


import seaborn as sns
class_names = [0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y = 1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# In[6]:


######################
##### PROBLEM 2 ######
######################

kfold = KFold(n_splits = 5, random_state = 0, shuffle = True)
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, X, Y, cv = kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[7]:


kfold = KFold(n_splits = 10, random_state = 0, shuffle = True)
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, X, Y, cv = kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[18]:


######################
#### PROBLEM 3.1 #####
######################

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer_data = cancer.data

Y = cancer.target

X_train, X_val, Y_train, Y_val = train_test_split(cancer_data, Y, train_size = 0.8, test_size = 0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)


# In[19]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_val)

cnf_matrix = confusion_matrix(Y_val, Y_pred)

print("Accuracy:", metrics.accuracy_score(Y_val, Y_pred))
print("Precision:", metrics.precision_score(Y_val, Y_pred))
print("Recall:", metrics.recall_score(Y_val, Y_pred))


# In[20]:


import seaborn as sns
class_names = [0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y = 1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# In[29]:


######################
#### PROBLEM 3.2 #####
######################

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer_data = cancer.data

Y = cancer.target

X_train, X_val, Y_train, Y_val = train_test_split(cancer_data, Y, train_size = 0.8, test_size = 0.2)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_val_std = scaler.transform(X_val)


# In[30]:


C = [10, 1, 0.1, 0.001]

for c in C:
    clf = LogisticRegression(penalty = 'l1', C = c, solver = 'liblinear')
    clf.fit(X_train, Y_train)
    print('C:', c)
    print('Training Accuracy:', clf.score(X_train_std, Y_train))
    print('Test Accuracy:', clf.score(X_val_std, Y_val))
        
    Y_pred = clf.predict(X_val)

    cnf_matrix = confusion_matrix(Y_val, Y_pred)

    print("Precision:", metrics.precision_score(Y_val, Y_pred))
    print("Recall:", metrics.recall_score(Y_val, Y_pred))
    print('')
    


# In[13]:


######################
#### PROBLEM 4.1 #####
######################

kfold = KFold(n_splits = 5, random_state = 0, shuffle = True)
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, cancer_data, Y, cv = kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[14]:


kfold = KFold(n_splits = 10, random_state = 0, shuffle = True)
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, cancer_data, Y, cv = kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[26]:


######################
#### PROBLEM 4.2 #####
######################

C = [10, 1, 0.1, 0.001]

for c in C:
    kfold = KFold(n_splits = 5, random_state = 0, shuffle = True)
    model = LogisticRegression(penalty = 'l1', C = c, solver = 'liblinear')
    results = cross_val_score(model, cancer_data, Y, cv = kfold)
    print('C:', c)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[ ]:




