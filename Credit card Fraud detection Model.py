#!/usr/bin/env python
# coding: utf-8

# # Credit Card online fraud detection model-Using Python

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Uploading the dataset

# In[4]:


data = pd.read_csv("C:/Users/sweta/Downloads/Data/Credit Card.csv")
print(data.head())


# ## finding the null values

# In[5]:


print(data.isnull().sum())


# In[6]:


# Exploring transaction type
print(data.type.value_counts())


# In[7]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values


# In[8]:


import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()


# ## correlation between the features of the data with the isFraud column:

# In[9]:


# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# ## transforming categorical features into numerical features

# In[10]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# ## training a model

# In[11]:


# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[12]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)


# In[13]:


model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[14]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))

