# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAJAN P
RegisterNumber: 212223240121 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
Date set:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/e64b690a-a621-49b8-9932-cb8749114809)

Head values:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/d08c3406-436b-4c0e-b926-4f70f48a69f7)

Tail values:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/5f8bbc8f-ee3e-478e-8317-5fda86fcc307)

X and y values:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/43f37992-27e2-4afe-abf7-df03d3c785ca)

Predication values of X and Y:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/0a1e76b7-df2c-412e-9e79-c9b9f46bad9b)

MAE,MSE and RMSE:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/2792ba2d-2f31-4f16-8e7e-801543c83a25)

Training Set:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/fb80334a-24d1-4106-8e2e-f0f0cc011d96)

Testing Set:
![image](https://github.com/PRAJAN-23013995/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150313345/22e0e674-a75c-4239-96fe-662524dc3514)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
