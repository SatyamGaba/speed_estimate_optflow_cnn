#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error


# In[10]:


y_true = np.array(pd.read_csv("train.txt", header=None, squeeze=True))
y_pred = np.array(pd.read_csv("test.txt", header=None, squeeze=True))


# In[11]:


mean_squared_error(y_true, y_pred)


# In[ ]:




