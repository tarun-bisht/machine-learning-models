# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:27:04 2019

@author: Tarun Bisht
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("Data.csv")
features=dataset.iloc[:,:-1].values
label=dataset.iloc[:,-1].values

#taking care of null data using mean strategy
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
features[:,1:]=imputer.fit_transform(features[:,1:])

#Encoding Categorical data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_x=LabelEncoder()
features[:,0]=label_encoder_x.fit_transform(features[:,0])
one_hot_encoder=OneHotEncoder(categorical_features=[0])
features=one_hot_encoder.fit_transform(features).toarray()
label_encoder_y=LabelEncoder()
label[:]=label_encoder_y.fit_transform(label[:])

#Splitting Training and Test Data
from sklearn.model_selection import train_test_split
train_feature,test_feature,train_label,test_label=train_test_split(features,label,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_feature=sc_x.fit_transform(train_feature)
test_feature=sc_x.transform(test_feature)
