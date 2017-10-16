
# coding: utf-8

# In[1]:

import sys
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn import ensemble, preprocessing
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

##Read the data
train_df=pd.read_csv('/Users/shashank/Desktop/csvs/train_pCWxroh.csv')
test_df=pd.read_csv('/Users/shashank/Desktop/csvs/test_bKeE5T8.csv')


# In[3]:

# Drop the Christmas trend
train_df=train_df.drop(train_df.index[[1834,1835,1836,1837,1838,1839,1840,1841]])


# In[5]:

# Test ids to include in final submission csv
ids=list(test_df['ID'])


# In[6]:

# Convert the date string to datetime
train_df['ID']=train_df['ID'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))
test_df['ID']=test_df['ID'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))


# In[7]:

# Convert the datetime to timestamp
train_df["Date"] = (pd.to_datetime(train_df["ID"], format="%Y-%m-%d %H:%M"))
test_df["Date"] = (pd.to_datetime(test_df["ID"], format="%Y-%m-%d %H:%M"))


# In[8]:

train_y = np.array(train_df.Count.values)
test_id = test_df.ID.values


# In[9]:

# Extract Features
train_df["Year"] = train_df["Date"].apply(lambda x: x.year)
test_df["Year"] = test_df["Date"].apply(lambda x: x.year)
train_df["Hour"] = train_df["Date"].apply(lambda x: x.hour)
test_df["Hour"] = test_df["Date"].apply(lambda x: x.hour)
train_df["WeekDay"] = train_df["Date"].apply(lambda x: x.weekday())
test_df["WeekDay"] = test_df["Date"].apply(lambda x: x.weekday())
train_df["DayCount"] = train_df["Date"].apply(lambda x: x.toordinal())
test_df["DayCount"] = test_df["Date"].apply(lambda x: x.toordinal())


# In[10]:

train = train_df.drop(["ID","Date","Count"], axis=1)
test = test_df.drop(["ID","Date"], axis=1)


# In[11]:

## "One hot encoding.."
temp_train_arr = np.empty([train.shape[0],0])
temp_test_arr = np.empty([test.shape[0],0])
cols_to_drop = []


# In[12]:

for var in train.columns:
    if var in ["Hour", "WeekDay"]:
        print var
        lb = preprocessing.LabelEncoder()
        full_var_data = pd.concat((train[var],test[var]),axis=0).astype('str')
        temp = lb.fit_transform(np.array(full_var_data))
        train[var] = lb.transform(np.array( train[var] ).astype('str'))
        test[var] = lb.transform(np.array( test[var] ).astype('str'))
        cols_to_drop.append(var)
        ohe = preprocessing.OneHotEncoder(sparse=False)
        ohe.fit(temp.reshape(-1,1))
        temp_arr = ohe.transform(train[var].reshape(-1,1))
        temp_train_arr = np.hstack([temp_train_arr, temp_arr])
        temp_arr = ohe.transform(test[var].reshape(-1,1))
        temp_test_arr = np.hstack([temp_test_arr, temp_arr])


# In[13]:

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)
train = np.hstack( [np.array(train),temp_train_arr]).astype("float64")
test = np.hstack( [np.array(test),temp_test_arr]).astype("float64")


# In[14]:

# Training on only latest Data
train_X = np.array(train)[12000:]
train_y = train_y[12000:]
test_X = np.array(test)
test_X=np.delete(test_X, 0, axis=1)


# In[15]:

reg = lm.LinearRegression()
reg.fit(train_X, train_y)
preds = reg.predict(test_X).astype('int')


# In[16]:

lr = pd.DataFrame({"predicted":preds})
lr.to_csv('/Users/shashank/Desktop/LR_predictions.csv',index=False)


# In[ ]:



