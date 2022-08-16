# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 15:44:58 2022

@author: agraw
"""

import pandas as pd 
import numpy as np
df = pd.read_csv("laptop_data.csv")
df.columns
df.shape
df = df.drop(['Unnamed: 0'],axis=1)
df.isnull().sum()
df.dtypes
df.duplicated().sum()
df.head(5)
fd = df.drop_duplicates()
fd.duplicated().sum()
fd
fd['Ram'] = fd['Ram'].str.replace('GB','')
fd.head(2)
fd['Ram']
fd['Weight'] = fd['Weight'].str.replace('kg','')
fd['Ram']=fd['Ram'].astype('int32')
fd['Weight'] = fd['Weight'].astype('float32')
fd.info()

import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['Price'])

# to plot number of laptops for particular company
fd['Company'].value_counts().plot(kind='bar')
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

fd['TypeName'].value_counts().plot(kind='bar')

sns.barplot(x=fd['TypeName'],y=fd['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(fd['Inches'])
sns.scatterplot(x=fd['Inches'],y=fd['Price'])

#to add column name touchscreen
fd['Touchscreen']=fd['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

fd.head()
fd['Touchscreen'].value_counts().plot(kind='bar')

sns.barplot(x=fd['Touchscreen'],y=fd['Price'])
plt.xticks(rotation='horizontal')
plt.show()


fd['IPS']=fd['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
fd['IPS'].value_counts().plot(kind='bar')
sns.barplot(x=fd['IPS'],y=fd['Price'])


new = fd['ScreenResolution'].str.split('x',n=1,expand = True)
fd['X_res'] = new[0]
fd['Y_res'] = new[1]

fd.head()

#to convert into number
fd['X_res']=fd['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
fd.head()

fd.dtypes
fd['Y_res']
fd['X_res'] = fd['X_res'].astype('int32')
fd['Y_res'] = fd['Y_res'].astype('int32')

#correlation
fd.corr()['Price']
df.columns

#New column PPI
fd['ppi']=(((fd['X_res']**2)+(fd['Y_res']**2))**0.5/fd['Inches']).astype('float')
fd.corr()['Price']

#remove columns
fd= fd.drop(['ScreenResolution'],axis=1)

fd.columns
fd= fd.drop(['Inches','X_res','Y_res'],axis=1)
fd.columns

fd['Cpu Name'] = fd['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
fd.head()


#function 
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
        
        
fd['Cpu brand'] = fd['Cpu Name'].apply(fetch_processor)
fd['Cpu brand'].value_counts().plot(kind='bar')

sns.barplot(x=fd['Cpu brand'],y=fd['Price'])
plt.xticks(rotation='vertical')
plt.show()

fd.drop(columns=['Cpu','Cpu Name'],inplace=True)
fd.columns

fd['Ram'].value_counts().plot(kind='bar')

sns.barplot(x=fd['Ram'],y=fd['Price'])
plt.xticks(rotation='vertical')
plt.show()

fd['Memory'].value_counts()


# To convert memory column into 4 diffrent columns namely SSD, HDD, Flash storage ,Hybrid
#following code of 

fd['Memory'] = fd['Memory'].astype(str).replace('\.0', '', regex=True)
fd["Memory"] = fd["Memory"].str.replace('GB', '')
fd["Memory"] = fd["Memory"].str.replace('TB', '000')
new = fd["Memory"].str.split("+", n = 1, expand = True)

fd["first"]= new[0]
fd["first"]=fd["first"].str.strip()

fd["second"]= new[1]

fd["Layer1HDD"] = fd["first"].apply(lambda x: 1 if "HDD" in x else 0)
fd["Layer1SSD"] = fd["first"].apply(lambda x: 1 if "SSD" in x else 0)
fd["Layer1Hybrid"] = fd["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
fd["Layer1Flash_Storage"] = fd["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

fd['first'] = fd['first'].str.replace(r'\D', '')

fd["second"].fillna("0", inplace = True)

fd["Layer2HDD"] = fd["second"].apply(lambda x: 1 if "HDD" in x else 0)
fd["Layer2SSD"] = fd["second"].apply(lambda x: 1 if "SSD" in x else 0)
fd["Layer2Hybrid"] = fd["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
fd["Layer2Flash_Storage"] = fd["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

fd['second'] = fd['second'].str.replace(r'\D', '')

fd["first"] = fd["first"].astype(int)
fd["second"] = fd["second"].astype(int)

fd["HDD"]=(fd["first"]*fd["Layer1HDD"]+fd["second"]*fd["Layer2HDD"])
fd["SSD"]=(fd["first"]*fd["Layer1SSD"]+fd["second"]*fd["Layer2SSD"])
fd["Hybrid"]=(fd["first"]*fd["Layer1Hybrid"]+fd["second"]*fd["Layer2Hybrid"])
fd["Flash_Storage"]=(fd["first"]*fd["Layer1Flash_Storage"]+fd["second"]*fd["Layer2Flash_Storage"])

fd.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)



# till here the code is given 


fd.drop(columns=['Memory'],inplace=True)

fd.corr()['Price']
fd.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

fd.head()

fd['Gpu'].value_counts()
# to exract brand name
fd['Gpu brand'] = fd['Gpu'].apply(lambda x:x.split()[0])

fd['Gpu brand'].value_counts()
# we will remove arm as it is only 1 

fd = fd[fd['Gpu brand'] != 'ARM']

sns.barplot(x=fd['Gpu brand'],y=fd['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

fd.drop(columns=['Gpu'],inplace=True)

#OPERATING SYSTEM COLUMN
fd['OpSys'].value_counts()

sns.barplot(x=fd['OpSys'],y=fd['Price'])
plt.xticks(rotation='vertical')
plt.show()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

fd['os'] = fd['OpSys'].apply(cat_os)

fd.drop(columns=['OpSys'],inplace=True)

sns.distplot(fd['Weight'])
fd.corr()['Price']

sns.heatmap(fd.corr())

#dark color more correlation

sns.distplot(np.log(df['Price']))

X = fd.drop(columns=['Price'])
y=  np.log(fd['Price'])

X
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

x_train.dtypes





