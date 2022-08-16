# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:25:50 2022

@author: agraw
"""

import pandas as pd 
df = pd.read_csv("college.csv")
df.columns
df.describe
df.dtypes
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
df['type_school']=Le.fit_transform(df['type_school'])
df['school_accreditation']=Le.fit_transform(df['school_accreditation'])
df['gender'] = Le.fit_transform(df['gender'])
df['interest'] = Le.fit_transform(df['interest'])
df['residence'] = Le.fit_transform(df['residence'])

x = df.drop(['in_college'],axis=1).values
y =df['in_college'].values
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5)





model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,pred)
