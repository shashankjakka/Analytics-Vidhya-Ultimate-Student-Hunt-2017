
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd


# In[5]:

lr=pd.read_csv('/Users/shashank/Desktop/LR_predictions.csv')
xgb=pd.read_csv('/Users/shashank/Downloads/xgb_predcitions.csv')


# In[12]:

subs=0.5*lr["predicted"] + 0.5*xgb["predictions"]


# In[13]:

subs.to_csv('final.csv',index=False)

