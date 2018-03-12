
# coding: utf-8

# In[199]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('data/train.csv')


# In[200]:


data


# In[201]:


lb_make = LabelEncoder()
data["job_group_code_index"] = lb_make.fit_transform(data["job_group_code"])
data["job_code_index"] = lb_make.fit_transform(data["job_code"])
data["union_code_index"] = lb_make.fit_transform(data["union_code"])
data["salary_index"] = lb_make.fit_transform(data["salary"])
data["department_index"] = lb_make.fit_transform(data["department_name"])


# In[202]:


data


# In[203]:


data.dtypes


# In[204]:


data=data.drop('worker_group_name',1)
data=data.drop('department_code',1)
data=data.drop('department_name',1)
data=data.drop('union_name',1)
data=data.drop('job_group_code',1)
data=data.drop('job_group',1)
data=data.drop('job_code',1)
data=data.drop('job',1)
data=data.drop('union_code',1)
data=data.drop('id',1)
data=data.drop('salary',1)


# In[205]:


data.dtypes


# In[206]:


#model = LogisticRegression()
# create the RFE model and select 3 attributes
#X=data.drop('salary_index',1)
#rfe = RFE(model, 3)
#rfe = rfe.fit(X, data.salary_index)
# summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)


# In[207]:


data.isnull().sum()


# In[208]:


#import sklearn.cross_validation
#X=data.drop('salary',1)
#X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, data.salary, test_size=0.33, random_state = 5)


# In[209]:


lm = linear_model.LinearRegression()
X=data.drop('salary_index',1)
# fit our model!
model = lm.fit(X,data.salary_index)
# make predictions from our X variable using our model!

#predictions = lm.predict(X)


# In[210]:


lm.score(,data.salary_index)


# In[211]:


#plt.plot(X, predictions, color='blue', linewidth=3)
#plt.show()


# In[212]:


#print(predictions)[0:5]


# In[213]:


data2=pd.read_csv('data/test.csv')
lb_make = LabelEncoder()
data2["job_group_code_index"] = lb_make.fit_transform(data2["job_group_code"])
data2["job_code_index"] = lb_make.fit_transform(data2["job_code"])
data2["union_code_index"] = lb_make.fit_transform(data2["union_code"])
data2["department_index"] = lb_make.fit_transform(data2["department_name"])


# In[214]:


data2=data2.drop('worker_group_name',1)
data2=data2.drop('department_code',1)
data2=data2.drop('department_name',1)
data2=data2.drop('union_name',1)
data2=data2.drop('job_group_code',1)
data2=data2.drop('job_group',1)
data2=data2.drop('job_code',1)
data2=data2.drop('job',1)
data2=data2.drop('union_code',1)
data2=data2.drop('id',1)
data2=data2.drop('salary',1)


# In[215]:


data2.dtypes


# In[216]:


data2


# In[217]:


predictions = lm.predict(data2)
print(predictions)[0:5]


# In[218]:


print(predictions)


# In[219]:


lm.coef_


# In[220]:


lm.intercept_


# In[221]:


data4=pd.read_csv('data/test.csv')



# In[222]:


dataf = {'id': data4.id, 'salary': predictions}


# In[223]:


df = pd.DataFrame(data=dataf)


# In[226]:


df


# In[230]:


df.to_csv("result2.csv", sep=',')

