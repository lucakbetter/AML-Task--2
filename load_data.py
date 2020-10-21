import numpy as np
import pandas as pd

from pathlib import Path


def nanInfo(df, name):
    print("number of NaN's for each column in %s:\n%s\n" % (name, df.isnull().sum(axis = 0)))
    df_size = df.size
    df_nan_size = df.isnull().sum().sum()
    if df_nan_size != 0:
        nan_percentage = df_size/df_nan_size
        print('percentage of missing data in %s:\t%s\n' % (name, int(nan_percentage)/100))
    else:
        print('percentage of missing data in %s:\t%s\n' % (name, str(0)))
        
def countValues(y_pred, string):
    print('\nAnalysis of %s:' % string)
    unique, frequency = np.unique(y_pred, return_counts = True) 
    print("Values:",  unique) 
    print("Frequency:", frequency)


def dataframes():
    xTrainPath = Path('data', 'X_train.csv')
    yTrainPath = Path('data', 'y_train.csv')
    xTestPath = Path('data', 'X_test.csv')
    
    assert xTrainPath.exists() and yTrainPath.exists() and xTestPath.exists(), 'Wrong path.'
    
    x_train = pd.read_csv(xTrainPath, index_col=0)
    y_train = pd.read_csv(yTrainPath, index_col=0)
    x_test = pd.read_csv(xTestPath, index_col=0)
    
    x_train.index = x_train.index.astype(int)
    y_train.index = y_train.index.astype(int)
    x_test.index = x_test.index.astype(int)
    
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values.reshape(-1)
    
    return x_train, y_train, x_test

if __name__ == "__main__":
	main()
