#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Claire Ardern 
# Intro to Machine Learning - Homework 0 
# Linear Regression with Gradient Descent
# 09/20/2022


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# PROBLEM 1

# Read in data file.
df = pd.read_csv('./D3.csv')

# Create an array for each data set.
X_1 = df.values[:,0]
X_2 = df.values[:,1]
X_3 = df.values[:,2]
Y = df.values[:,3]

# Get length of data sets.
m = len(Y)

X_0 = np.ones((m,1))

# Reshape arrays for 2D formatting.
X_1 = X_1.reshape(m,1)
X_2 = X_2.reshape(m,1)
X_3 = X_3.reshape(m,1)

# Stack horizontally
X_1 = np.hstack((X_0, X_1))
X_2 = np.hstack((X_0, X_2))
X_3 = np.hstack((X_0, X_3))

# Initialize theta to zeros.
theta = np.zeros(2)


# In[4]:


# Loss Function

def compute_loss(X, Y, theta):
    
    predictions = X.dot(theta)
    errors = np.subtract(predictions, Y)
    sqrErrors = np.square(errors)
    J = 1 / (2*m) * np.sum(sqrErrors)
    
    return J
    


# In[5]:


# Computations for loss for each data set. 

loss1 = compute_loss(X_1, Y, theta)
print('The loss for given values of theta_0 and theta_1 using data set X_1:', loss1)

loss2 = compute_loss(X_2, Y, theta)
print('The loss for given values of theta_0 and theta_1 using data set X_2:', loss2)

loss3 = compute_loss(X_3, Y, theta)
print('The loss for given values of theta_0 and theta_1 using data set X_3:', loss3)


# In[6]:


# Gradient Descent Function

def gradient_descent(X, Y, theta, alpha, iterations):
    
        loss_history = np.zeros(iterations)
        
        for i in range(iterations): 
            
            predictions = X.dot(theta)
            errors = np.subtract(predictions, Y)
            sum_delta = (alpha / m) * X.transpose().dot(errors);
            theta = theta - sum_delta;
            loss_history[i] = compute_loss(X, Y, theta)
            
        return theta, loss_history


# In[7]:


# Computations for X_1 Dataset

theta = np.zeros(2)
iterations = 300
alpha = 0.1

theta, loss_history1 = gradient_descent(X_1, Y, theta, alpha, iterations)
print('Final values of theta for dataset X_1: ', theta)

plt.scatter(X_1[:,1], Y, color = 'red', marker = '+', label = 'Training Data')
plt.plot(X_1[:,1], X_1.dot(theta), color = 'green', label = 'Linear Regression')

plt.grid()
plt.xlabel('X_1 Dataset')
plt.ylabel('Predicted Outputs')
plt.title('Linear Regression for X_1 Dataset')
plt.legend()

plt.figure()
plt.plot(range(1, iterations+1), loss_history1, color = 'blue', label = "Loss over Iterations")

plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Calculated Loss')
plt.title('Loss over Iterations for X_1 Dataset')
plt.legend()


# In[8]:


# Computations for X_2 Dataset

theta = np.zeros(2)
iterations = 2000
alpha = 0.1

theta, loss_history2 = gradient_descent(X_2, Y, theta, alpha, iterations)
print('Final values of theta for dataset X_2: ', theta)
          
plt.scatter(X_2[:,1], Y, color = 'red', marker = '+', label = 'Training Data')
plt.plot(X_2[:,1], X_2.dot(theta), color = 'green', label = 'Linear Regression')

plt.grid()
plt.xlabel('X_2 Dataset')
plt.ylabel('Predicted Outputs')
plt.title('Linear Regression for X_2 Dataset')
plt.legend()     

plt.figure()
plt.plot(range(1, iterations+1), loss_history2, color = 'blue', label = "Loss over Iterations")

plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Calculated Loss')
plt.title('Loss over Iterations for X_2 Dataset')
plt.legend()


# In[9]:


# Computations for X_3 Dataset

theta = np.zeros(2)
iterations = 500
alpha = 0.1

theta, loss_history3 = gradient_descent(X_3, Y, theta, alpha, iterations)
print('Final values of theta for dataset X_3: ', theta)

plt.scatter(X_3[:,1], Y, color = 'red', marker = '+', label = 'Training Data')
plt.plot(X_3[:,1], X_3.dot(theta), color = 'green', label = 'Linear Regression')

plt.grid()
plt.xlabel('X_3 Dataset')
plt.ylabel('Predicted Outputs')
plt.title('Linear Regression for X_3 Dataset')
plt.legend() 

plt.figure()
plt.plot(range(1, iterations+1), loss_history3, color = 'blue', label = "Loss over Iterations")

plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Calculated Loss')
plt.title('Loss over Iterations for X_3 Dataset')
plt.legend()


# In[10]:


# PROBLEM 2

# Create an array for each data set.
X_1 = df.values[:,0]
X_2 = df.values[:,1]
X_3 = df.values[:,2]
Y = df.values[:,3]

# Get length of data sets.
m = len(Y)

X_0 = np.ones((m,1))

# Reshape arrays for 2D formatting.
X_1 = X_1.reshape(m,1)
X_2 = X_2.reshape(m,1)
X_3 = X_3.reshape(m,1)

# Stack horizontally
X = np.hstack((X_0, X_1, X_2, X_3))

# Initialize theta to zeros.
theta = np.zeros(4)


# In[11]:


iterations = 1200
alpha = 0.1

loss = compute_loss(X, Y, theta)
print('The loss for given values of theta using data set X:', loss)

theta, loss_history = gradient_descent(X, Y, theta, alpha, iterations)
print('Final values of theta for dataset X: ', theta)


# In[12]:


plt.figure()
plt.plot(range(1, iterations+1), loss_history, color = 'blue', label = "Loss over Iterations")

plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Calculated Loss')
plt.title('Loss over Iterations for X Dataset')
plt.legend()


# In[13]:


# Predict value of Y for new (X_1, X_2, X_3) values (1, 1, 1)

Y = theta[0]*1 + theta[1]*1 + theta[2]*1 + theta[3]
print('The predicted value of Y is ', Y)


# In[14]:


# Predict value of Y for new (X_1, X_2, X_3) values (2, 0, 4)

Y = theta[0]*2 + theta[1]*0 + theta[2]*4 + theta[3]
print('The predicted value of Y is ', Y)


# In[15]:


# Predict value of Y for new (X_1, X_2, X_3) values (3, 2, 1)

Y = theta[0]*3 + theta[1]*2 + theta[2]*1 + theta[3]
print('The predicted value of Y is ', Y)


# In[ ]:




