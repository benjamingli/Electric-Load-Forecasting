#-*- coding:utf-8 -*-
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import numpy as np
import sys

def dofile(filename):
    df = pd.read_csv(filename, index_col=['ID'])
    X = [] ; Y = []
    for i in range(df.shape[0]):
        x = df.loc[i,['site','year','month','day','hour',\
                'holiday','maxtemp','mintemp','weatherType']].tolist()
        y = df.loc[i,['load']].tolist()
        #标签取绝对值
        y = np.fabs(y)
        X.append(x)
        Y.append(y)
    #特征值归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_after = min_max_scaler.fit_transform(X)
    df_x = DataFrame(X_after)
    df_y = DataFrame(Y)
    new_data = np.concatenate((df_x,df_y), axis=1)
    df_new_data = DataFrame(new_data, columns=['site','year','month','day','hour',\
                                'holiday','maxtemp','mintemp','weatherType','load'])
    df_new_data.to_csv('/home/songling/load_ANN/pretrate.csv')

if __name__ == '__main__':
    dofile(sys.argv[1])
'''

    for i in range(start, end):
        x = df.loc[i,['site','year','month','day','hour',\
                'holiday','maxtemp','mintemp','weatherType']].tolist()
        y = df.loc[i,['load']].tolist()
        X.append(x)
        Y.append(y)
    #特征值归一化    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_after = min_max_scaler.fit_transform(X)

    return X_after,Y
'''
