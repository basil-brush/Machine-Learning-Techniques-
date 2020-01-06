
import pandas as pd
import math
import sklearn.svm
import sklearn.metrics as metrics
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split 
import seaborn as sns

missing_values = ["NaN"]

df = pd.read_csv("/Users/harryrodger/Desktop/COMPASS4DATA/data/Part2classification/adultTrain.csv", na_values=missing_values)
df1 = pd.read_csv("/Users/harryrodger/Desktop/COMPASS4DATA/data/Part2classification/adultTest.csv", na_values=missing_values)

#will give me an idea with what I should replace missing values with
#print(df.describe())
df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

df['nativeCountry'] = df['nativeCountry'].replace([' Cuba',' Jamaica',' India',' Mexico',' South',' Puerto-Rico',' Honduras',' England',' Canada',' Germany',' Iran',' Philippines',' Italy',' Poland',' Columbia',' Cambodia',' Thailand',' Ecuador',' Laos',' Haiti',' Taiwan',' Portugal',' Dominican-Republic',' El-Salvador',' France',' Guatemala',' China',' Japan',' Yugoslavia',' Peru',' Outlying-US(Guam-USVI-etc)',' Scotland',' Trinadad&Tobago',' Greece',' Nicaragua',' Vietnam',' Hong',' Ireland',' Hungary',' Holand-Netherlands'],'Non-US')

#Replacing all missing data for each features / note that not all features were treated the same in the sense of using the mean or using the mode...
df['workclass'].fillna("Private", inplace = True)
df['occupation'].fillna("Prof-speciality", inplace = True)
df['nativeCountry'].fillna("No Country", inplace = True)


df1['workclass'].fillna("Private", inplace = True)
df1['occupation'].fillna("Prof-speciality", inplace = True)
df1['nativeCountry'].fillna("No Country", inplace = True)

#Enconde the data so that it can be used for classification
df['workclass'] = df['workclass'].astype('category').cat.codes
df['marital-status'] = df['marital-status'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['relationship'] = df['relationship'].astype('category').cat.codes
df['race'] = df['race'].astype('category').cat.codes
df['sex'] = df['sex'].astype('category').cat.codes
df['nativeCountry'] = df['nativeCountry'].astype('category').cat.codes
df['salary'] = df['salary'].astype('category').cat.codes

df1['workclass'] = df['workclass'].astype('category').cat.codes
df1['marital-status'] = df['marital-status'].astype('category').cat.codes
df1['occupation'] = df['occupation'].astype('category').cat.codes
df1['relationship'] = df['relationship'].astype('category').cat.codes
df1['race'] = df['race'].astype('category').cat.codes
df1['sex'] = df['sex'].astype('category').cat.codes
df1['nativeCountry'] = df['nativeCountry'].astype('category').cat.codes
df1['salary'] = df['salary'].astype('category').cat.codes

print(df.head(21))


#Building the classifier

clf = LinearDiscriminantAnalysis()

X_train = df[['age','workclass', 'educationYears','marital-status','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry']]
y_train = df['salary']

X_test = df1[['age','workclass','educationYears','marital-status','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry']]
y_test = df1['salary']

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

#Printing results

print(confusion_matrix(y_test,y_pred))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Report : ')
print(classification_report(y_test, y_pred))



print('AUC:')
print(roc_auc_score(y_test,y_pred))

print("R2:")
print(r2_score(y_test,y_pred))