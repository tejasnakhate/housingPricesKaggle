# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:24:18 2021

@author: TNakhate
"""

import pandas as pd
import numpy as np
import os
import math
import json

os.chdir('C:/Users/TNakhate/Documents/Predictions/housingPricesKaggle/house prices')
#x = pd.read_csv('x.csv')
#y = pd.read_csv('y.csv')
test = pd.read_csv('test.csv')
test = test.drop(['Alley','PoolQC','Fence','MiscFeature'], axis=1)#dropped because of too much NA values
train = pd.read_csv('train.csv')
train = train.drop(['Alley','PoolQC','Fence','MiscFeature'], axis=1)#dropped because of too much NA values

#x = train[train.columns[~train.isnull().any()]]
#x_test = test[test.columns[~test.isnull().any()]]
 
#train['LotFrontage'].fillna(value=df['LotFrontage'].mean(axis=0),inplace=True)
#train['MasVnrArea'].fillna(value=df['MasVnrArea'].mean(axis=0),inplace=True)
#train['GarageYrBlt'].fillna(value=df['GarageYrBlt'].mean(axis=0),inplace=True)



# obj_df["Street"].dtypes

#one hot encoding
#pd.get_dummies(obj_df, columns=["MSZoning"], prefix=["MSZoning"]).head()
#codes encoding
#obj_df["MSZoningcode"] = obj_df["MSZoning"].cat.codes
#obj_df["MSZoningcode"] = obj_df["MSZoning"].cat.codes
#encoded_X = obj_df["Street"]
#for i in obj_df.columns:
#    if i != "Street" :
#        obj_df[i]=obj_df[i].astype('category')
#        encoded_X[i+"CODE"] = obj_df[i].cat.codes
        
#count unique values in df column
'''y.MasVnrType.unique()
y['Electrical'].value_counts()
y.nunique()
y['Electrical'].isna().sum()
electrical_cleanup = {"Electrical":{"SBrkr":13, "FuseA":5, "FuseF":4, "FuseP":2, "Mix":1}}
y = y.replace(electrical_cleanup)
y['Electrical'] = y['Electrical'].fillna(13)'''


def prepare_data_onehot_encoding(df, intersect):
    x = df[df.columns[~df.isnull().any()]]
    y = df[df.columns[df.isnull().any()]]
    
    oth_df = x.select_dtypes(include=['int64','float64']).copy()
    obj_df = x.select_dtypes(include=['object']).copy()
    obj_cols = set(obj_df.columns)
    remaining_cols = obj_cols.difference(intersect)
    onehot=pd.get_dummies(obj_df, columns=intersect, prefix=intersect)
    for col in remaining_cols:
        cleanup_json = get_cleanup_json(col, obj_df)
        onehot = onehot.replace(cleanup_json)
    final_x = pd.concat([onehot, oth_df], axis=1)    
    return final_x,y

def prepare_data_categorical_encoding(df):
    x = df[df.columns[~df.isnull().any()]]
    y = df[df.columns[df.isnull().any()]]
    
    oth_df = x.select_dtypes(include=['int64','float64']).copy()
    obj_df = x.select_dtypes(include=['object']).copy()
    obj_cols = set(obj_df.columns)
    for col in obj_cols:
        cleanup_json = get_cleanup_json(col, obj_df)
        obj_df = obj_df.replace(cleanup_json)
    final_x = pd.concat([obj_df, oth_df], axis=1)    
    return final_x,y

def get_cleanup_json(col_name, y):
    count_na = y[col_name].value_counts().sort_values(ascending=True)
    indexes = list(count_na.index)
    cleanup_json = {}
    cleanup_json[col_name] = {}
    j = 1
    for i in range(0, len(indexes)):
        cleanup_json[col_name][indexes[i]] = j
        j += 1
    return cleanup_json


def predict_na(col_name, x, y, predicted, cleanup=None):
    if cleanup is not None:
        y = y.replace(cleanup)
    train_x = x[~y[col_name].isna()]
    train_x = pd.concat([train_x, predicted.loc[~y[col_name].isna()].reindex(train_x.index)], axis=1)
    train_y = y[~y[col_name].isna()][col_name].to_frame()
    test_x = x[y[col_name].isna()]
    test_x = pd.concat([test_x, predicted.loc[y[col_name].isna()].reindex(test_x.index)], axis=1)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)
    test_y = clf.predict(test_x)
    j = 0
    for i in y.index:
        if math.isnan(float(y[col_name][i])):
            y[col_name][i] = test_y[j]
            # print(y[col_name][i])
            j += 1
    predicted = pd.concat([predicted, y[col_name]], axis=1)
    y = y.drop([col_name], axis=1)
    return predicted
'''
trainobj = train.select_dtypes(include=['object']).copy()
trainobj = trainobj[trainobj. columns[~trainobj. isnull(). any()]]
testobj = test.select_dtypes(include=['object']).copy()
testobj = testobj[testobj. columns[~testobj. isnull(). any()]]
traincols = set(trainobj.columns)
testcols = set(testobj.columns)
intersect = traincols.intersection(testcols)
x,y = prepare_data_onehot_encoding(test, intersect)
'''

def remove_na(x,y):
    na_counts = y.isna().sum().sort_values(ascending=True)
    cols = list(na_counts.index)
    predicted = pd.DataFrame()
    
    for i in range(1,len(cols)):
        if str(y[cols[i]].dtypes) == 'object':
            cleanup_json = get_cleanup_json(cols[i], y)
            predict = predict_na(cols[i], x, y, predicted, cleanup_json)
        else:
            predict = predict_na(cols[i], x, y, predicted)
        predicted = predict
        y = y.drop([cols[i]], axis=1)
    return predicted

x,y = prepare_data_categorical_encoding(train)
x1,y1 = prepare_data_categorical_encoding(test)
predicted = remove_na(x,y)     
no_na = pd.concat([x, predicted.reindex(x.index)], axis = 1)
no_na = no_na.drop(['Id'], axis = 1)
no_na = no_na.loc[:, ~no_na.columns.str.contains('^Unnamed')]
final_train_x = no_na.drop(['SalePrice'], axis = 1)
final_train_y = no_na['SalePrice'].to_frame()

predicted = remove_na(x1, y1)
final_test_x = pd.concat([x1, predicted.reindex(x1.index)], axis = 1)
final_test_x = final_test_x.drop(['Id'], axis = 1)
final_test_x = final_test_x.loc[:, ~final_test_x.columns.str.contains('^Unnamed')]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(final_train_x, final_train_y)

y_pred = classifier.predict(final_test_x)
pd.DataFrame(y_pred).to_csv("predictions.csv")
