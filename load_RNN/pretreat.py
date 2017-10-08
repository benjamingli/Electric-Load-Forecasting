#-*- coding:utf-8 -*-
import pandas as pd
from sklearn import preprocessing

def dofile(filename,datasize):
    df = pd.read_csv(filename, index_col=['ID'])
    X = [] ; Y = []
    for i in range(datasize):
        x = df.loc[i,['hour','holiday','maxtemp','mintemp']].tolist()
        y = df.loc[i,['load']].tolist()
        X.append(x)
        Y.append(y)
    return X,Y
