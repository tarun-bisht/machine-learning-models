# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:55:34 2019

@author: Tarun Bisht
"""
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")
features=dataset.iloc[:,:-1].values
label=dataset.iloc[:,-1].values

#Splitting Training and Test Data
from sklearn.model_selection import train_test_split
train_feature,test_feature,train_label,test_label=train_test_split(features,label,test_size=0.2,random_state=0)

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(train_feature,train_label)

#Visualizing Linear Regression
##Training Visualization
print("Training Data Visualization")
train_predict=linear_regressor.predict(train_feature)
plt.scatter(train_feature,train_label,color="red")
plt.plot(train_feature,train_predict,color="blue")
plt.title("Salary Vs Experience ")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
##Test Visualization
print("Test Data Prediction Visualization")
test_predict=linear_regressor.predict(test_feature);
plt.scatter(test_feature,test_label,color="red")
plt.plot(test_feature,test_predict,color="blue")
plt.title("Salary Vs Experience ")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Show Performance using R Square Method
from sklearn.metrics import r2_score
print("R Squared value (Training) := ",r2_score(test_label,test_predict))
print("R square value (Test):= ",r2_score(train_label,train_predict))

##Predicting data
try:
    exp=eval(input("Enter your Experience: "))
    salary=linear_regressor.predict([[exp]])
    print("Salary Predicted By Machine: ",salary)
except Exception as e:
    print("ERROR:::: ",e)

