#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# ## Predict survival on the Titanic
# 
# 1]Defining the problem statement
# 
# 2]Collecting the data
# 
# 3]Exploratory data analysis
# 
# 4]Feature engineering
# 
# 5]Modelling
# 
# 6]Testing

# # 1. Defining the problem statement
# Complete the analysis of what sorts of people were likely to survive.
#  In particular, we ask you to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

# In[279]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")


# # 2. Collecting the data
# training data set and testing data set are given by Kaggle you can download from
#  kaggle directly 
#  
#  link-https://www.kaggle.com/c/titanic/data

# ## load train, test dataset using Pandas

# In[280]:


import pandas as pd

train = pd.read_csv('G:\\data\\kaggle_titanic\\train.csv')
test = pd.read_csv('G:\\data\\kaggle_titanic\\test.csv')


# # 3. Exploratory data analysis
# Printing first 5 rows of the train dataset.

# In[281]:


train.head()


# # Details of the Data
# -survived: 0 = No, 1 = Yes
# 
# -pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# -sibsp: no. of siblings / spouses aboard the Titanic
# 
# -parch: no.  of parents / children aboard the Titanic
# 
# -ticket: Ticket number
# 
# -cabin: Cabin number
# 
# -embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# 
# ps-The following details was avaliable in kaggle website

# In[282]:


train.shape


# number of passangers details avaliable for training-891
# 
# number of avaiable features/columns/details -12

# In[283]:


test.shape


# number of passangers details avaliable for training-418
# 
# number of avaiable features/columns/details -11(except for the result/survial/target column)

# In[284]:


train.info()


# We can see that Age value is missing for many rows.
# 
# Out of 891 rows, the Age value is present only in 714 rows.
# 
# Similarly, Cabin values are also missing in many rows. Only 204 out of 891 rows have Cabin values.

# In[285]:


test.info()


# We can see that Age value is missing for many rows.
# 
# Out of 418 rows, the Age value is present only in 332 rows.
# 
# Similarly, Cabin values are also missing in many rows. Only 418 out of 91 rows have Cabin values.

# In[286]:


train.isnull().sum()


# In[287]:


test.isnull().sum()


# ## visualization of Data

# In[288]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# ## Bar Chart for Categorical Features
# -Pclass
# 
# -Sex
# 
# -SibSp ( # of siblings and spouse)
# 
# -Parch ( # of parents and children)
# 
# -Embarked
# 
# -Cabin

# In[289]:


survived = train[train['Survived']==1]["Pclass"].value_counts()
dead = train[train['Survived']==0]["Pclass"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# 
# **The Chart confirms 1st class more likely survivied than other classes**
# 
# **The Chart confirms 3rd class more likely dead than other classes**

# In[290]:


survived = train[train['Survived']==1]["Sex"].value_counts()
dead = train[train['Survived']==0]["Sex"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
#  **The Chart confirms Women more likely survivied than Men**

# In[291]:


survived = train[train['Survived']==1]["SibSp"].value_counts()
dead = train[train['Survived']==0]["SibSp"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# **The Chart confirms a person aboarded with more than 2 siblings or spouse more likely survived**
# 
# **The Chart confirms a person aboarded without siblings or spouse more likely dead**

# In[292]:


survived = train[train['Survived']==1]["Parch"].value_counts()
dead = train[train['Survived']==0]["Parch"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# 
# **The Chart confirms a person aboarded with more than 2 parents or children more likely survived**
# 
# **The Chart confirms a person aboarded alone more likely dead**

# In[293]:


survived = train[train['Survived']==1]["Embarked"].value_counts()
dead = train[train['Survived']==0]["Embarked"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# 
# **The Chart confirms a person aboarded from C slightly more likely survived**
# 
# **The Chart confirms a person aboarded from Q more likely dead**
# 
# **The Chart confirms a person aboarded from S more likely dead**

# # 4. Feature engineering
# Feature engineering is the process of using domain knowledge of the data
# 
# to create features that make machine learning algorithms work.

# In[294]:


train.head()


# In[295]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[296]:


train['Title'].value_counts()


# In[297]:


test['Title'].value_counts()


# ### Title map
# Mr : 0
# 
#  Miss : 1
#  
#   Mrs: 2
#   
#    Others: 3

# In[298]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[299]:


train.head()


# In[300]:


survived = train[train['Survived']==1]["Title"].value_counts()
dead = train[train['Survived']==0]["Title"].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# 
# **-The Chart confirms a person  with title Miss more likely survived**
# 
# **-The Chart confirms a person with title Mr more likely Dead**
# 
# **-The Chart confirms a person  with  title Mrs  more likely survived**

# In[301]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# ###  Sex Mapping 
# 
# Male-0
# 
# Female-1

# In[302]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# ### Dealing with Missing ages
# 
# **As a lot of ages are missing we will replace it with median ages of their respective title**

# In[303]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# The other way to implement  the above code is to convert dataframe into 2d array then iterate over two for loops if for
# the first for loop has a value of 0 then take corresponding age and add it to a list this way we get list of ages for
# particular title them take median of the list and use fill.na() to put that value

# In[304]:


train.head()


# In[305]:


train.info()


# **Converting Numerical Age to Categorical Variable**
# 
#  mapping  done as follows :
#  
# child: 0
# 
# young: 1
# 
# adult: 2
# 
# mid-age: 3
# 
# senior: 4

# In[306]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[307]:


train.head()


# In[308]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ### Insights gained
# **more than 50% of 1st class are from S embark**
# 
# **more than 85% of 2nd class are from S embark**
# 
# **more than 70% of 3rd class are from S embark**

# In[309]:


#fill out missing embark with S embark
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[310]:



train.head()


# In[311]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[312]:


train.info()


# In[313]:


train.isna().sum()


# In[314]:


test.isna().sum()


# In[315]:


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[316]:


test.isna().sum()


# In[317]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[318]:


train.head()


# In[319]:


train['Cabin'].value_counts()


# In[320]:


train.Cabin.unique()


# In[321]:


train.info()


# In[322]:


test.info()


# In[323]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[324]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[325]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[326]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[327]:


train.isna().sum()


# In[328]:


test.isna().sum()


# **We have filled all the missing values**

#         

# Now we will add an extra feature called family size which is the sum of no. of siblings/spouse (sibsp) and no. of parents/children (parch) 

# In[329]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[330]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[331]:


train.head()


# Now we will drop some features which are useless for the prediction and sibsp and parch as we have created and extra feature using thoose two and also the passenger id just in training dataset as it is useless for the result

# In[332]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[333]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[334]:


train_data.head(10)


# # 5. Modelling

# In[339]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
import numpy as np


# ## Cross Validation (K-fold)

# In[340]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[341]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data,target,test_size=0.2,random_state=42)


# ## kNN

# In[342]:


# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5,  p = 2)
knn_classifier.fit(x_train, y_train)
y_pred_knn = knn_classifier.predict(x_test)
accuracy_score(y_test, y_pred_knn)


# ## Random forest

# In[343]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred_rf)


# ## logistic regression

# In[345]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state =42)
lr_classifier.fit(x_train, y_train)
y_pred_lr = lr_classifier.predict(x_test)
accuracy_score(y_test, y_pred_lr)


# ## Descision tree

# In[346]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(x_train, y_train)
y_pred_dt = dt_classifier.predict(x_test)
accuracy_score(y_test, y_pred_dt)


# ## Support vector machine

# In[349]:


#support vector machine
from sklearn import svm
from sklearn import metrics
cls=svm.SVC(kernel="linear")
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
metrics.accuracy_score(y_test,pred)


# ## Xgboost

# In[353]:


#xgboost algorithm
import xgboost as xgb
xgb=xgb.XGBClassifier()
xgb.fit(x_train,y_train)
pred_xgb=xgb.predict(x_test)
accuracy_score(y_test, pred_xgb)


# ## Bagging

# In[355]:


#Bagging 
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 40)
bg.fit(x_train,y_train)
bg.score(x_test,y_test)


# In[356]:


#Bagging 

bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(x_train,y_train)
bg.score(x_test,y_test)


# ## Ada boost

# In[357]:


#Boosting - Ada Boost

adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
adb.fit(x_train,y_train)

adb.score(x_train,y_train)


# In[358]:


adb.score(x_test,y_test)


# ## Voting classifier 

# In[366]:


# Voting Classifier - Multiple Model Ensemble 
import xgboost as xgb 
xgb = xgb.XGBClassifier()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 2 )


# In[367]:


evc = VotingClassifier( estimators= [('xgb',xgb),('dt',dt),('svm',svm)], voting = 'hard')


# In[376]:


evc.fit(x_train,y_train)


# In[377]:


evc.score(x_test, y_test)


# # Testing

# In[370]:


test_data = test.drop("PassengerId", axis=1).copy()
prediction = evc.predict(test_data)


# In[372]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[375]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})
output.to_csv('my_submission_2.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




