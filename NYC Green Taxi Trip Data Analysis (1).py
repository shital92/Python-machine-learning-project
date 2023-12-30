#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\kaila\\OneDrive\\Desktop\\Datasets"')


# In[7]:


greentaxi=pd.read_parquet("green_tripdata_2023-05.parquet",engine="pyarrow")
# pyarrow is library for importing parquet files


# In[13]:


greentaxi.info()


# In[12]:


greentaxi=greentaxi.drop("ehail_fee",axis=1) # Drop Null Column


# In[19]:


# Create a new variable called "trip_duration"
greentaxi["trip_duration"]=greentaxi.lpep_dropoff_datetime-greentaxi.lpep_pickup_datetime


# In[22]:


greentaxi.trip_duration.head()


# In[21]:


# dt refers to date time Library in pandas
greentaxi.trip_duration=greentaxi.trip_duration.dt.total_seconds()/60


# In[28]:


# Extract New variable "weekday" from datetime
greentaxi["weekday"]=greentaxi.lpep_dropoff_datetime.dt.weekday


# In[29]:


greentaxi.weekday.value_counts(dropna=False)
# Monday -0 & Sunday -6


# In[30]:


# Extract New variable "hourofday" from datetime
greentaxi["hourofday"]=greentaxi.lpep_dropoff_datetime.dt.hour


# In[31]:


greentaxi.hourofday.value_counts(dropna=False)


# In[44]:


greentaxi.isnull().sum().sort_values(ascending=False)


# In[35]:


greentaxi.trip_type.value_counts(dropna=False)


# In[34]:


greentaxi.trip_type=greentaxi.trip_type.fillna(1.0)


# In[39]:


greentaxi.store_and_fwd_flag.value_counts(dropna=False)


# In[38]:


greentaxi.store_and_fwd_flag=greentaxi.store_and_fwd_flag.fillna("N")


# In[43]:


greentaxi.RatecodeID.value_counts(dropna=False)


# In[42]:


greentaxi.RatecodeID=greentaxi.RatecodeID.fillna(1.0)


# In[47]:


greentaxi.congestion_surcharge.describe()


# In[46]:


greentaxi.congestion_surcharge=greentaxi.congestion_surcharge.fillna(
    greentaxi.congestion_surcharge.median())


# In[50]:


greentaxi.passenger_count.value_counts(dropna=False)


# In[49]:


greentaxi.passenger_count=greentaxi.passenger_count.fillna(1.0)


# In[53]:


greentaxi.payment_type.value_counts(dropna=False)


# In[52]:


greentaxi.payment_type=greentaxi.payment_type.fillna(1.0)


# In[54]:


greentaxi.columns


# In[55]:


# Pie Diagrams of - trip_type, payment_type, RatecodeID
# Stacked Bar Diagram of - weekday and trip_type, weekday & payment_type, hour and payment_type
# groupby() mean for following: weekday & total_amount, hour & total_amount, payment_type &
# total_amount
# groupby() mean for following: weekday & trip_duration, hour & trip_duration, payment_type &
# trip_duration
# Hypothesis Testing
# Test Null Average total_amount of different trip_type is equal
# Test Null Average tip_amount of different trip_type is equal

# Test Null Average total_amount for different weekday equal
# Test Null Average tip_amount for different weekday equal

# Test Null No Association between trip_type and payment_type
# Test Null No Association between RatecodeID and payment_type
# Test Null No Association between weekday and payment_type
# Test Null No Association between weekday and trip_type


# In[56]:


greentaxi.trip_type.value_counts().plot(kind="pie",autopct="%.2f%%")


# In[57]:


greentaxi.payment_type.value_counts().plot(kind="pie",autopct="%.2f%%")


# In[58]:


greentaxi.RatecodeID.value_counts().plot(kind="pie",autopct="%.2f%%")


# In[59]:


ax=pd.crosstab(greentaxi.weekday,greentaxi.trip_type).plot(kind="bar",stacked=True)
for i in ax.containers:
    ax.bar_label(i)


# In[60]:


ax=pd.crosstab(greentaxi.weekday,greentaxi.payment_type).plot(kind="bar",stacked=True)
for i in ax.containers:
    ax.bar_label(i)


# In[64]:


plt.figure(figsize=(30,15))
ax=pd.crosstab(greentaxi.hourofday,greentaxi.payment_type).plot(kind="bar",stacked=True)
for i in ax.containers:
    ax.bar_label(i)


# In[67]:


ax=greentaxi.total_amount.groupby(greentaxi.weekday).mean().plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[73]:


plt.figure(figsize=(20,10))
ax=np.round(greentaxi.total_amount.groupby(
    greentaxi.hourofday).mean(),2).sort_values(ascending=False).plot(kind="bar",color="red")
for i in ax.containers:
    ax.bar_label(i)


# In[75]:


ax=greentaxi.total_amount.groupby(greentaxi.payment_type).mean().plot(kind="bar",color="violet")
for i in ax.containers:
    ax.bar_label(i)


# In[76]:


ax=greentaxi.trip_duration.groupby(greentaxi.weekday).mean().plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[78]:


plt.figure(figsize=(20,10))
ax=np.round(greentaxi.trip_duration.groupby(
    greentaxi.hourofday).mean(),2).sort_values(ascending=False).plot(kind="bar",color="green")
for i in ax.containers:
    ax.bar_label(i)


# In[79]:


ax=greentaxi.trip_duration.groupby(greentaxi.payment_type).mean().plot(kind="bar",color="pink")
for i in ax.containers:
    ax.bar_label(i)


# In[80]:


# Hypothesis Test process
# groupby() mean & var, Null & Alt, Split Data, Conduct Test, Infer p-value


# In[81]:


# Test Null Average total_amount of different trip_type is equal
greentaxi.total_amount.groupby(greentaxi.trip_type).mean() # Exactly 2 Levels 


# In[82]:


greentaxi.total_amount.groupby(greentaxi.trip_type).var()


# In[83]:


# Null - There is no Significant difference in Average total_amount of trip_type 1 & 2.
# Alt - There is Significant difference in Average total_amount of trip_type 1 & 2.


# In[84]:


# Split Data 
trip_type_1=greentaxi[greentaxi.trip_type==1.0]
trip_type_2=greentaxi[greentaxi.trip_type==2.0]


# In[85]:


from scipy.stats import ttest_ind


# In[86]:


ttest_ind(trip_type_1.total_amount,trip_type_2.total_amount,equal_var=False)
# Since pvalue=5.457304023638798e-60 is less than 0.05, Reject Null Hypothesis


# In[87]:


greentaxi.tip_amount.groupby(greentaxi.trip_type).mean()


# In[88]:


greentaxi.tip_amount.groupby(greentaxi.trip_type).var()


# In[90]:


# Null - There is no Significant difference in Average tip_amount of trip_type 1 & 2.
# Alt - There is Significant difference in Average tip_amount of trip_type 1 & 2.


# In[89]:


ttest_ind(trip_type_1.tip_amount,trip_type_2.tip_amount,equal_var=False)
# Since  pvalue=0.001074 is less than 0.05, Reject Null


# In[91]:


greentaxi.total_amount.groupby(greentaxi.weekday).mean()


# In[92]:


mon=greentaxi[greentaxi.weekday==0]
tue=greentaxi[greentaxi.weekday==1]
wed=greentaxi[greentaxi.weekday==2]
thu=greentaxi[greentaxi.weekday==3]
fri=greentaxi[greentaxi.weekday==4]
sat=greentaxi[greentaxi.weekday==5]
sun=greentaxi[greentaxi.weekday==6]


# In[95]:


# Null - There is no significant difference in Average total_amount for different weekday
# Alt - There is significant difference in Average total_amount for different weekday


# In[93]:


from scipy.stats import f_oneway


# In[94]:


f_oneway(mon.total_amount,tue.total_amount,wed.total_amount,thu.total_amount,fri.total_amount,
        sat.total_amount,sun.total_amount)
# Since pvalue=1.5781665738843715e-07 is less than 0.05, Reject Null


# In[97]:


greentaxi.tip_amount.groupby(greentaxi.weekday).mean()


# In[98]:


# Null - There is no significant difference in Average tip_amount for different weekday
# Alt - There is significant difference in Average tip_amount for different weekday


# In[99]:


f_oneway(mon.tip_amount,tue.tip_amount,wed.tip_amount,thu.tip_amount,fri.tip_amount,
        sat.tip_amount,sun.tip_amount)
# Since pvalue=1.1035304312725308e-05 is less than 0.05, Reject Null


# In[100]:


pd.crosstab(greentaxi.trip_type,greentaxi.payment_type)


# In[101]:


from scipy.stats import chi2_contingency


# In[102]:


# Null - There is no association between both variables
# Alt - There is association between both variables


# In[103]:


chi2_contingency(pd.crosstab(greentaxi.trip_type,greentaxi.payment_type))
# Since pvalue=0.0004385 is less than 0.05, Reject Null


# In[104]:


pd.crosstab(greentaxi.RatecodeID,greentaxi.payment_type)


# In[105]:


# Null - There is no association between both variables
# Alt - There is association between both variables


# In[106]:


chi2_contingency(pd.crosstab(greentaxi.RatecodeID,greentaxi.payment_type))
# Since  pvalue=3.2670230054629802e-43 is less than 0.05, Reject Null


# In[107]:


pd.crosstab(greentaxi.weekday,greentaxi.payment_type)


# In[108]:


# Null - There is no association between both variables
# Alt - There is association between both variables


# In[109]:


chi2_contingency(pd.crosstab(greentaxi.weekday,greentaxi.payment_type))
# Since  pvalue=4.532248995318973e-08 is less than 0.05, Reject Null


# In[110]:


pd.crosstab(greentaxi.trip_type,greentaxi.weekday)


# In[111]:


# Null - There is no association between both variables
# Alt - There is association between both variables


# In[113]:


chi2_contingency(pd.crosstab(greentaxi.trip_type,greentaxi.weekday))
# Since pvalue=3.458694063594574e-39 is less than 0.05, Reject Null


# In[114]:


greentaxi.columns


# In[115]:


# Split Data into numeric and objectols
numericcols=greentaxi[['trip_distance', 'fare_amount', 'extra', 'mta_tax',
       'tip_amount', 'tolls_amount', 'improvement_surcharge','congestion_surcharge',
                      'trip_duration']]


# In[116]:


objectcols=greentaxi[['store_and_fwd_flag', 'RatecodeID','passenger_count','payment_type',
                      'trip_type','weekday', 'hourofday']]


# In[117]:


numericcols.head()


# In[118]:


objectcols.columns


# In[119]:


objectcols_dummy=pd.get_dummies(objectcols,columns=['store_and_fwd_flag', 'RatecodeID', 
                                                    'passenger_count', 'payment_type',
                                                    'trip_type', 'weekday', 'hourofday'])


# In[120]:


X=pd.concat([numericcols,objectcols_dummy],axis=1)


# In[121]:


y=greentaxi.total_amount


# In[122]:


import seaborn as sns


# In[123]:


# Histogram, boxplot & Density Curve - y
plt.figure(figsize=(30,15))
fig,ax=plt.subplots(3,1)
sns.kdeplot(y,ax=ax[0])
sns.boxplot(y,orient="h",ax=ax[1])
sns.histplot(y,ax=ax[2])
plt.show()


# In[124]:


plt.figure(figsize=(30,15))
fig,ax=plt.subplots(3,1)
sns.kdeplot(np.log(y),ax=ax[0])
sns.boxplot(np.log(y),orient="h",ax=ax[1])
sns.histplot(np.log(y),ax=ax[2])
plt.show()


# In[130]:


plt.figure(figsize=(20,10))
sns.heatmap(numericcols.corr(),annot=True)


# In[133]:


numericcols2=numericcols


# In[134]:


numericcols2['total_amount']=greentaxi.total_amount


# In[135]:


plt.figure(figsize=(20,10))
sns.heatmap(numericcols2.corr(),annot=True)


# In[136]:


X=X.drop('fare_amount',axis=1)


# In[137]:


from sklearn.linear_model import LinearRegression


# In[138]:


reg=LinearRegression()


# In[143]:


regmodel=reg.fit(X,y)


# In[144]:


regmodel.score(X,y)


# In[145]:


regpredict=regmodel.predict(X)


# In[146]:


regresid=y-regpredict


# In[147]:


np.sqrt(np.mean(regresid**2)) # RMSE


# In[148]:


greentaxi.total_amount.describe()


# In[ ]:




