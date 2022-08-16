# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:28:18 2022

@author: agraw
"""

import pandas as pd
df = pd.read_csv('Airlines.csv')
df.dtypes
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
Le= LabelEncoder()
df['Airline'] = df['Airline'].fillna('N')
df['Airline']=Le.fit_transform(df['Airline'])
df.dtypes
df['AirportFrom'] = Le.fit_transform(df['AirportFrom'])
df['AirportTo'] = Le.fit_transform(df['AirportTo'])
df.dtypes

x=df.drop(['Delay'],axis=1).values
y=df['Delay'].values

import seaborn as sns
sns.heatmap(df,annot=False,cmap = 'Blues')

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,pred)
