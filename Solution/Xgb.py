
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from sklearn.metrics import r2_score,mean_squared_error
import time
import datetime
from math import sqrt
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import cross_val_score
import xgboost as xgb


# In[3]:


##Read the data
train_df=pd.read_csv('train_pCWxroh.csv')
test_df=pd.read_csv('test_bKeE5T8.csv')


# In[4]:


# Drop the Christmas trend
train_df=train_df.drop(train_df.index[[1834,1835,1836,1837,1838,1839,1840,1841]])


# In[5]:


# Convert the date string to datetime
train_df['ID']=train_df['ID'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))
test_df['ID']=test_df['ID'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))


# In[6]:


# Convert the datetime to timestamp
train_df["Date"] = (pd.to_datetime(train_df["ID"], format="%Y-%m-%d %H:%M"))
test_df["Date"] = (pd.to_datetime(test_df["ID"], format="%Y-%m-%d %H:%M"))


# In[7]:


train_y = np.array(train_df.Count.values)
test_id = test_df.ID.values


# In[8]:


# Extract the features
train_df["Year"] = train_df["Date"].apply(lambda x: x.year)
test_df["Year"] = test_df["Date"].apply(lambda x: x.year)
train_df["Hour"] = train_df["Date"].apply(lambda x: x.hour)
test_df["Hour"] = test_df["Date"].apply(lambda x: x.hour)
train_df["WeekDay"] = train_df["Date"].apply(lambda x: x.weekday())
test_df["WeekDay"] = test_df["Date"].apply(lambda x: x.weekday())
train_df["DayCount"] = train_df["Date"].apply(lambda x: x.toordinal())
test_df["DayCount"] = test_df["Date"].apply(lambda x: x.toordinal())


# In[9]:


train = train_df.drop(["ID","Date","Count"], axis=1)
test = test_df.drop(["ID","Date"], axis=1)
test.drop(['Count'],axis=1,inplace=True)


# In[12]:


def runXGB(train_X, train_y, test_X, test_y=None):
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.02
        params["min_child_weight"] = 8
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["seed"] = 1
        plst = list(params.items())
        num_rounds = 500

        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y = model.predict(xgtest)
        return pred_test_y


# In[13]:


preds = runXGB(np.array(train), train_y, np.array(test))


# In[18]:


xgb=pd.DataFrame({'predictions':preds})


# In[19]:


xgb.to_csv('xgb_predcitions.csv',index=False)

