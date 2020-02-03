import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


mar_budget    = np.array([[60], [80],  [100], [30], [50], [20], [90],  [10], [20], [120]],  dtype=float)
subs_gained = np.array([[160], [200], [240], [100], [140], [80], [220], [60], [80], [280]],  dtype=float)

def display_plot():
    plt.scatter(mar_budget, subs_gained)
    plt.xlim(0,120)
    plt.ylim(0,260)
    plt.xlabel('Marketing Budget(in thousand of Dollars)')
    plt.ylabel('Subscribers Gained(in thousand)')
    plt.show()

def split_train_test_and_train_model():
    X_train, X_test, y_train, y_test = train_test_split(mar_budget, subs_gained, test_size=0.2, random_state=0)
    global regressor
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) #training the algorithm

def output():
    value = int(input("Enter the value to predict: "))
    if(value):
        y_pred = regressor.predict([[value]])
        print(y_pred)

if __name__ == "__main__":
    split_train_test_and_train_model()
    output()
    