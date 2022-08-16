# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:26:01 2022

@author: agraw
"""

import pandas as pd
df = pd.read_csv('basket.csv')
df.head()
df.dtypes
df
df1=df.drop(['InvoiceNo'],axis=1)
def encode_units(x):
    if x<=0:
        return 0
    else:
        return 1
    
basket_sets = df1.applymap(encode_units)
basket_sets
from efficient_apriori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



Frequent_itemsets = apriori(basket_sets, min_support = 0.02, use_colnames=True)
Frequent_itemsets
Res = association_rules(Frequent_itemsets, metric='confidence', min_threshold = 0.0001)
print(Res)
df_out = Res[['antecedents','consequents','confidence','support','lift']]

df_out.to_csv("market.csv")
