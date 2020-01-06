import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

missing_values = ["NaN"]

df = pd.read_csv("/Users/harryrodger/Desktop/COMPASS4DATA/data2/Part2classification/adult.csv", na_values=missing_values)

print(df.describe())
print(df.info())

df['salary'] = df['salary'].astype('category').cat.codes

#Age exploring

sns.countplot(y="age", hue="salary",data=df)
plt.show()

#workclass exploring

sns.countplot(y="workclass", hue="salary",data=df)
plt.show()

print (df[['workclass','salary']].groupby(['workclass']).mean())

#race exploring

sns.countplot(y="race", hue="salary",data=df)
plt.show()

print (df[['race','salary']].groupby(['race']).mean())

#Education exploring

sns.countplot(y="educationYears", hue="salary",data=df)
plt.show()

print (df[['educationYears','salary']].groupby(['educationYears']).mean())

#marital-status exploring

print (df[['marital-status','salary']].groupby(['marital-status']).mean())

sns.countplot(y="marital-status", hue="salary",data=df)
plt.show()

df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

print (df[['marital-status','salary']].groupby(['marital-status']).mean())

print(df[['marital-status','relationship','salary']].groupby(['relationship','marital-status']).mean())

#nativeCountry exploring

print (df[['nativeCountry','salary']].groupby(['nativeCountry']).mean())

sns.countplot(y="nativeCountry", hue="salary",data=df)
plt.show()

df['nativeCountry'] = df['nativeCountry'].replace([' Cuba',' Jamaica',' India',' Mexico',' South',' Puerto-Rico',' Honduras',' England',' Canada',' Germany',' Iran',' Philippines',' Italy',' Poland',' Columbia',' Cambodia',' Thailand',' Ecuador',' Laos',' Haiti',' Taiwan',' Portugal',' Dominican-Republic',' El-Salvador',' France',' Guatemala',' China',' Japan',' Yugoslavia',' Peru',' Outlying-US(Guam-USVI-etc)',' Scotland',' Trinadad&Tobago',' Greece',' Nicaragua',' Vietnam',' Hong',' Ireland',' Hungary',' Holand-Netherlands'],'Non-US')

sns.countplot(y="nativeCountry", hue="salary",data=df)
plt.show()

print (df[['nativeCountry','salary']].groupby(['nativeCountry']).mean())

# relationship exploring

sns.countplot(y="relationship", hue="salary",data=df)
plt.show()

print (df[['relationship','salary']].groupby(['relationship']).mean())


# occupation exploring

sns.countplot(y="occupation", hue="salary",data=df)
plt.show()

print (df[['occupation','salary']].groupby(['occupation']).mean())


#Replacing all missing data for each features / note that not all features were treated the same in the sense of using the mean or using the mode...
df['workclass'].fillna("No Country", inplace = True)
df['occupation'].fillna("No Occupation", inplace = True)
df['nativeCountry'].fillna("No Country", inplace = True)

#Enconde the data so that it can be converted from categorical data for classification
df['workclass'] = df['workclass'].astype('category').cat.codes
df['marital-status'] = df['marital-status'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['relationship'] = df['relationship'].astype('category').cat.codes
df['race'] = df['race'].astype('category').cat.codes
df['sex'] = df['sex'].astype('category').cat.codes
df['nativeCountry'] = df['nativeCountry'].astype('category').cat.codes

corr = df.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)

corr = df.corr()
fig, ax = plt.subplots(figsize=(20,20))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)),corr.columns)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.show()




