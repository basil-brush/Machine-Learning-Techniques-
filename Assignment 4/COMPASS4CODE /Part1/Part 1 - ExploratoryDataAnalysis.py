# This is the code use for COMP 309 Assignment 4 Part 1 (2.1)

import pandas as pd
import sklearn.svm, sklearn.metrics
import seaborn as sns

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#ensures that the print function will print everything relevant
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#imports the diamond.csv for evaluation etc...
df = pd.read_csv("/Users/harryrodger/Desktop/COMPASS4DATA/data/Part1regression/diamonds.csv")

print(df.info())

print(df.describe())

df.hist()
plt.show()

plt.hist('depth' , data=df , bins=25)
plt.show()

corr = df.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)

sns.factorplot(x='color', y='price', data=df, kind='box' ,aspect=2.5)
plt.show()

#gives the number of rows and collumns (collumns, rows)
#print(df.shape)

#prints the attrbutes of the data set
#print(df.columns)

#checks for missing values
#print(df.isnull().values.any())

#prints the first five elements of the diamonds.csv file
#print(df.head())

#for column in df.columns:
 #   print(column, ":" ,df[column].unique(),"\n")

#gives interesting information regarding the data
#print(df.describe())

#histogram of all attributes
#df.hist()
#plt.show()

#gives the mean of each feature in relation to the price
#print(df.groupby("price").mean())


pd.crosstab(df.cut,df.color).plot(kind='bar')
plt.title('Price of diamond in relation to its depth')
plt.xlabel('Price')
plt.ylabel('Amount sold')
plt.savefig('Price in reltion to depth')
plt.show()

df.boxplot('price','depth', rot= 30,figsize=(7,8))
plt.savefig('Price in relation to depth')
plt.show()

df.plot(x='depth', y='price', style='o')  
plt.title('depth vs price')  
plt.xlabel('depth')  
plt.ylabel('price') 
plt.savefig('depth vs price') 
plt.show()