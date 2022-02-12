#!/usr/bin/env python
# coding: utf-8

# # CASESTUDY- 10 PROBABILITY

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 


# In[24]:


data=pd.read_csv(r'C:\Users\Zephyr\Documents\ICT Assignments and Casestudies\Casestudy#10\mushrooms.csv')
data.head()


# In[25]:


data.shape


# In[26]:


data.isna().sum()


# In[27]:


data.dtypes


# In[28]:


data.info()


# In[29]:


data.nunique()


# In[30]:


data['class'].value_counts(normalize=True)


# Almost equal number of poisonous and edible mushroom types

# # Pre Processsing

# In[31]:


data2=data.drop(['veil-type'],axis=1)
data2.head()


# Since veil has the same value so droppping it.

# In[32]:


data2.columns


# # Encoding

# In[33]:


# Performing dummy encoding on the features
x= pd.get_dummies(data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']])


# In[34]:


x


# In[35]:


# PERFORMING LABEL ENCODING ON THE TARGET COLUMN

from sklearn.preprocessing import LabelEncoder
en=LabelEncoder()
y=en.fit_transform(data2['class'])


# In[36]:


y


# In[37]:


#SPLITTING THE DATASET 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# # Performing Classification

# ## 1. Logistic Regression

# In[40]:


#Logistic Regression

from sklearn.linear_model import LogisticRegression
Reg = LogisticRegression()
Reg.fit(x_train, y_train)
y_pred = Reg.predict(x_test)


# In[41]:


#CHECKING THE PERFORMANCE

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# ## 2.KNN Classifier

# In[42]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
acc_values=[]
neighbors=np.arange(3,15)
for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    classifier.fit(x_train, y_train)
    k_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test, k_pred)
    acc_values.append(acc)


# In[43]:


acc_values


# In[44]:


plt.plot(neighbors, acc_values,'o-')
plt.xlabel('k Value')
plt.ylabel('Accuracy')


# Taking the value of k as 3.The data point is 11 and square root of 11 is 3.3( we take an odd no. value as 3)

# In[45]:


classifier=KNeighborsClassifier(n_neighbors=3, metric='minkowski')
classifier.fit(x_train, y_train)
k_pred=classifier.predict(x_test)


# In[46]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,k_pred))


# ## 3.Decision Tree Classifier

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()
dtree.fit(x_train, y_train)
d_pred= dtree.predict(x_test)


# In[48]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,d_pred))


# ## 4.SVM Classifier

# In[49]:


# Linear SVM
from sklearn.svm import SVC
linear = SVC(kernel='linear')
linear.fit(x_train, y_train)
s_pred= linear.predict(x_test)


# In[50]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,s_pred))


# In[51]:


#Radial based function SVM
radial = SVC(kernel='rbf')
radial.fit(x_train, y_train)
r_pred = radial.predict(x_test)


# In[52]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,r_pred))


# ## 5.Random Forest Classifier

# In[53]:



from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
a_pred=rf.predict(x_test)


# In[54]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,a_pred))


# ## 6.Gradient Boosting

# In[55]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
g_pred= gb.predict(x_test)


# In[56]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,g_pred))


# ## 7.Naive Bayes Classifer

# ### a) Bernoulli Naive Bayes Classifier

# In[57]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
b_pred = bnb.predict(x_test)


# In[58]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,b_pred))


# ### b) Multinomial Naive Bayes Classifier

# In[59]:


from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
mn.fit(x_train, y_train)
m_pred= mn.predict(x_test)


# In[60]:


#CHECKING THE PERFORMANCE

print(metrics.classification_report(y_test,m_pred))


# ## Result: 

# I carried out different classifications and have found out that all the classification models except Naive Bayes models
# have yielded a 100% accuracy. 
# 
# In the Naive Bayes classification model BernoulliNB classifier provided 94% accuracy and MultinomialNB classifier 
# provided 95% accuracy.
