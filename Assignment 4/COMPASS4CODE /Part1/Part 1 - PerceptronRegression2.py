
import pandas as pd
import sklearn.svm
import sklearn.metrics as metrics
import numpy as np
import seaborn as seabornInstance 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import time


df = pd.read_csv("/Users/harryrodger/Desktop/COMPASS4DATA/data/Part1regression/diamonds.csv")

#preprocessing
df['color'] = df['color'].astype('category').cat.codes
df['clarity'] = df['clarity'].astype('category').cat.codes
df['cut'] = df['cut'].astype('category').cat.codes

df = df[(df[['x','y','z']] != 0).all(axis=1)] #essentially remoevs the zero values by creating a new one without them

X = df[['carat','cut','color','clarity','depth','table','x','y']].values
y = df['price'].values

start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=309)

reg = Perceptron()
reg.fit(X_train,y_train)

"""
coeff_df = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient']) 
print(coeff_df)
"""

y_pred = reg.predict(X_test)

end = time.time()

prediction = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})

print(prediction)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2:")
print(r2_score(y_test,y_pred))

print('time:')
print(end - start)
"""
plt.scatter(X_test,y_test, color = 'grey')
plt.plot(X_test,y_pred, color='red', linewidth=2)
plt.show()

prediction.head(50).plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-', linewidth = '0.5', color = 'green')
plt.grid(which='minor',linestyle='-', linewidth = '0.5', color = 'black')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['price'])
plt.show()

"""
