# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:02:32 2022

@author: agraw
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn

df = pd.read_csv("Stores.csv")
df.columns
df.isnull().sum()


plt.plot(df['Store_Sales'])
sn.heatmap(df)


