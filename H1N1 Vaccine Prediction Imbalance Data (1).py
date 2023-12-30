#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\user\\Desktop\\flue vaccines"')


# In[3]:


vaccinetrain=pd.read_csv("training_set_features.csv")


# In[4]:


vaccinetest=pd.read_csv("test_set_features.csv")


# In[5]:


vaccinelabels=pd.read_csv("training_set_labels.csv")
# 2 Dependent Variables - seasonal_vaccine & h1n1_vaccine
# seasonal_vaccine - Balanced data
# h1n1_vaccine - Imabalanced Data


# In[6]:


# seasonal_vaccine - Balanced data
ax=vaccinelabels.seasonal_vaccine.value_counts().plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[7]:


# h1n1_vaccine - Imbalanced data
ax=vaccinelabels.h1n1_vaccine.value_counts().plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[8]:


vaccinetrain.info()


# In[9]:


vaccinetest.info()


# In[10]:


# Concatenate both dataframes for preprocessing
combinedf=pd.concat([vaccinetrain,vaccinetest],axis=0)


# In[11]:


# Algorithm Based Missing Value Imputation - Considers impact of
# other variables on the missing value and imputes accordingly.
# MICE - Multivariate Imputation using Chained Equations
# Imputes both numeric and non numeric categorical variables

# Step 1 - Impute mising vales with mean (Numeric) & mode (categorical)
# Step 2 - Makes the Missing values variable as dependent variable
# and if missing value variable is
# Numeric - Regression Algorithm
# Non Numeric - Classiciation Algorithm
# Chained Equations is where missing values variable/column is
# treated as dependent variable and all other variables as independent
# variables and relevant regression/classification model is built.
# Step3 - Missing Values will be replaced by predicted value/class


# ![image.png](attachment:image.png)

# In[12]:


# In Python IterativeImputer in sklearn is experimental as of now
# IterativeImputer is similat to MICE algorithm
# Before implementing IterativeImputer
# 1) Drop irrelevant variables or columns
# 2) LabelEncode all object and categrocial data, but retain 
# missing values as missing. 


# In[11]:


combinedf=combinedf.drop("respondent_id",axis=1)


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


original=combinedf


# In[14]:


mask=combinedf.isnull()


# In[15]:


combinedf=combinedf.astype(str).apply(LabelEncoder().fit_transform)


# In[16]:


combinedf=combinedf.where(~mask,original)


# In[17]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier


# In[18]:


imputer=IterativeImputer(estimator=DecisionTreeClassifier(),
                        initial_strategy="most_frequent",
                        max_iter=50)


# In[19]:


combinedf_impute=imputer.fit_transform(combinedf)


# In[20]:


combinedf_impute=pd.DataFrame(combinedf_impute,
                              columns=combinedf.columns)


# In[21]:


combinedf_impute.to_csv("vaccineimpute.csv")


# In[22]:


plt.figure(figsize=(20,10))
ax=combinedf.employment_occupation.value_counts(
    dropna=False).plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[23]:


plt.figure(figsize=(20,10))
ax=combinedf_impute.employment_occupation.value_counts(
    dropna=False).plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i)


# In[24]:


#split data back to train & test
vaccinetrain_df=combinedf_impute.loc[0:26706]
vaccinetest_df=combinedf_impute.loc[26707:53414]


# In[25]:


print(vaccinetrain_df.shape)
print(vaccinetest_df.shape)


# In[26]:


y=vaccinelabels.h1n1_vaccine
X=vaccinetrain_df


# In[27]:


pd.DataFrame(y).value_counts().plot(kind="bar") # Imbalance in Classes or
# Levels - 0 - Majority Class(21033) & 1- Minority Class(5674)


# In[28]:


y=LabelEncoder().fit_transform(y)


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[30]:


logit=LogisticRegression(max_iter=1000)


# In[31]:


logitmodel=logit.fit(X,y)


# In[32]:


logitmodel.score(X,y) # Accuracy


# In[33]:


logitpredict=logitmodel.predict(X)


# In[34]:


print(classification_report(y,logitpredict))
# Miroity Class (1) prediction scores are worst


# In[41]:


# Imbalance data is aproblem in classification both in binary and
# multinomial classification

# In Imbalance Data within dependent Variable, Majotity class will be more
# than 70% and Minority Class will be less than 30%.

# Model performance is severely effected particularly in case of Minority
# Class

# Overall Accuracy of the model and Majority Class precision, recall and
# F1 Score will be good but Minority class precision, recall and F1Score
# will be worst

# Even AUC will be bad for Imbalanced Data.

#Classification report must be checked in case of Imbalance data


# In[43]:


# Dealing with Imbalance data - Sampling must be used. 2 Types of Sampling
# 1) Random Oversampling - In this method observations from minority class
# are duplicated till it equals the majority class

# 2) Random Undersampling - In this method observations from majority class
# are deleted till it equals the minority class

# Oversampling techniques are used when data is smaller in size
# Undersampling techniques are used when data is larger in size

# Popular Oversampling Technique is SMOTE.
# Popular Undersampling Technique is Tomek Links


# ![image.png](attachment:image.png)

# In[35]:


from imblearn.under_sampling import RandomUnderSampler


# In[36]:


rus=RandomUnderSampler(random_state=42)


# In[37]:


X_rus,y_rus=rus.fit_resample(X,y)


# In[38]:


print(pd.DataFrame(y).value_counts())
print(pd.DataFrame(y_rus).value_counts())


# In[39]:


logit_rus_model=logit.fit(X_rus,y_rus)


# In[40]:


logit_rus_model.score(X_rus,y_rus)


# In[41]:


logit_rus_predict=logit_rus_model.predict(X_rus)


# In[42]:


print(classification_report(y_rus,logit_rus_predict))


# In[43]:


from sklearn.metrics import RocCurveDisplay


# In[44]:


RocCurveDisplay.from_predictions(y,logitpredict)


# In[45]:


RocCurveDisplay.from_predictions(y_rus,logit_rus_predict)


# In[46]:


from imblearn.over_sampling import RandomOverSampler


# In[47]:


ros=RandomOverSampler(random_state=42)


# In[48]:


X_ros,y_ros=ros.fit_resample(X,y)


# In[49]:


print(pd.DataFrame(y).value_counts())
print(pd.DataFrame(y_ros).value_counts())


# In[50]:


logit_ros_model=logit.fit(X_ros,y_ros)


# In[51]:


logit_ros_model.score(X_ros,y_ros)


# In[52]:


logit_ros_predict=logit_ros_model.predict(X_ros)


# In[53]:


print(classification_report(y_ros,logit_ros_predict))


# In[54]:


RocCurveDisplay.from_predictions(y_ros,logit_ros_predict)


# In[65]:


# Oversampling Techinque - SMOTE - Synthetic Minority Oversampling
# Technique
# SMOTE uses KNN Algorithm(Euclidean Distance) and creates artifical or
# synthetic data that lies within data range
# No Outliers are created
# SMOTE also uses Random number generator for generating random weights
# between 0 & 1

# Two Independent Variables - X1 - Income and X2 - Age
# X1 - 2400, 2500, 2700, 2300, 2100, 2440
# X2 - 46, 34, 45, 28, 25, 41

# Choose a random weight between 0 & 1. Randonly selected - 0.60
# 2500+0.60*(2400-2500) = 2440 (synthetic data point)
# 34 + 0.60*(46-34)=41 (synthetic  data point)

# Different Types of SMOTE
# SMOTE will only with Numeric data
# SMOTENC will work for both numeric and nonnumeric categorical data
# SMOTEN will work for only Nonnumeric Categrical Data


# ![image.png](attachment:image.png)

# In[55]:


from imblearn.over_sampling import SMOTEN


# In[56]:


smote=SMOTEN(random_state=42)


# In[57]:


X_smote,y_smote=smote.fit_resample(X,y)


# In[58]:


print(pd.DataFrame(y).value_counts())
print(pd.DataFrame(y_smote).value_counts())


# In[59]:


logit_smote_model=logit.fit(X_smote,y_smote)


# In[60]:


logit_smote_model.score(X_smote,y_smote)


# In[61]:


logit_smote_predict=logit_smote_model.predict(X_smote)


# In[62]:


print(classification_report(y_smote,logit_smote_predict))


# In[63]:


RocCurveDisplay.from_predictions(y_smote,logit_smote_predict)


# In[64]:


from sklearn.tree import DecisionTreeClassifier


# In[65]:


tree=DecisionTreeClassifier(max_depth=12)


# In[66]:


treemodel=tree.fit(X,y)


# In[67]:


treemodel.score(X,y)


# In[68]:


treepredict=treemodel.predict(X)


# In[69]:


print(classification_report(y,treepredict))


# In[70]:


tree_smote_model=tree.fit(X_smote,y_smote)


# In[71]:


tree_smote_model.score(X_smote,y_smote)


# In[72]:


tree_smote_predict=tree_smote_model.predict(X_smote)


# In[73]:


print(classification_report(y_smote,tree_smote_predict))


# In[74]:


RocCurveDisplay.from_predictions(y_smote,tree_smote_predict)


# In[75]:


from sklearn.ensemble import RandomForestClassifier


# In[76]:


RF=RandomForestClassifier(n_estimators=3000,max_depth=12)


# In[77]:


RFmodel=RF.fit(X,y)


# In[78]:


RFmodel.score(X,y)


# In[79]:


RFpredict=RFmodel.predict(X)


# In[80]:


print(classification_report(y,RFpredict))


# In[81]:


RF_smote_model=RF.fit(X_smote,y_smote)


# In[82]:


RF_smote_model.score(X_smote,y_smote)


# In[83]:


RF_smote_predict=RF_smote_model.predict(X_smote)


# In[84]:


print(classification_report(y_smote,RF_smote_predict))


# In[85]:


from sklearn.ensemble import GradientBoostingClassifier


# In[86]:


gbm=GradientBoostingClassifier(n_estimators=1000)


# In[87]:


gbmmodel=gbm.fit(X,y)


# In[88]:


gbmmodel.score(X,y)


# In[89]:


gbmpredict=gbmmodel.predict(X)


# In[90]:


print(classification_report(y,gbmpredict))


# In[91]:


gbm_smote_model=gbm.fit(X_smote,y_smote)


# In[92]:


gbm_smote_model.score(X_smote,y_smote)


# In[93]:


gbm_smote_predict=gbm_smote_model.predict(X_smote)


# In[94]:


print(classification_report(y_smote,gbm_smote_predict))


# In[95]:


RocCurveDisplay.from_predictions(y_smote,gbm_smote_predict)


# In[96]:


test_gbm=gbm_smote_model.predict_proba(vaccinetest_df)


# In[97]:


pd.DataFrame(test_gbm).to_csv("gbm_h1n1.csv")


# In[98]:


test_RF=RF_smote_model.predict_proba(vaccinetest_df)


# In[99]:


pd.DataFrame(test_RF).to_csv("RF_h1n1.csv")


# In[100]:


test_logit=logit_smote_model.predict_proba(vaccinetest_df)


# In[134]:


pd.DataFrame(test_logit).to_csv("logit_h1n1.csv")


# In[ ]:




