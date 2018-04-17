
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Combining both the train and test dataset to do feature engineering. We will divide them later.

# In[4]:


train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, data.shape)


# In[5]:


data.head()


# We will now check for the missing values.

# In[6]:


data.apply(lambda x: sum(x.isnull()))


# Lets look at some basic statistics for numerical variables.

# In[7]:


data.describe()


# In[8]:


data.apply(lambda x: len(x.unique()))


# Let’s explore further using the frequency of different categories in each nominal variable.

# In[9]:


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['id', 'source', 'stroke']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())


# In[10]:


sns.countplot(x = 'smoking_status', hue= 'gender', data= data)


# ## Data Cleaning

# ### Imputing values for smoking status

# In[11]:


def impute_smoking_status(cols):
    smoking_status = cols[0]
    work_type = cols[1]
    hypertension = cols[2]
    age = cols[3]
    
    if pd.isnull(smoking_status):
        
        if work_type == 'children' or hypertension == 1:
            return 'never smoked'
        elif work_type!= 'children' and hypertension == 0:
            if age>=50:
                return 'formerly smoked'
            elif age<50:
                return 'smokes'
    else:
        return smoking_status
        


# In[12]:


data['smoking_status'] = data[['smoking_status', 'work_type', 'hypertension', 'age']].apply(impute_smoking_status,
                                                                                           axis= 1)

data['smoking_status'].fillna(value= 'never smoked', inplace= True)
# Again checking for the frequency of different categories.

# In[13]:


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['id', 'source', 'stroke']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())


# ### Imputing values for bmi.

# In[14]:


def impute_bmi(cols):
    
    Gender = cols[0]
    bmi = cols[1]
    smoking_status = cols[2]
    
    if pd.isnull(bmi):
        if Gender == 'Female':
            if smoking_status == 'formerly smoked':
                return 30.460
            elif smoking_status == 'never smoked':
                return 27.744
            elif smoking_status == 'smokes':
                return 29.360
        elif Gender == 'Male':
            if smoking_status == 'formerly smoked':
                return 30.732
            elif smoking_status == 'never smoked':
                return 26.535
            elif smoking_status == 'smokes':
                return 29.492
        else:
            return 26.538
    else:
        return bmi


# In[15]:


data['bmi'] = data[['gender', 'bmi', 'smoking_status']].apply(impute_bmi, axis = 1)


# In[16]:


data.info()


# ## Feature Engineering

# In[17]:


def diabetic(col):
    Glucose = col[0]
    
    if Glucose < 145:
        return 0
    else:
        return 1


# In[18]:


data['diabetic'] = data[['avg_glucose_level']].apply(diabetic, axis= 1)


# In[19]:


data.head()


# In[20]:


def Health(col):
    
    bmi = col[0]
    
    if bmi < 18.5:
        return 'underweight'
    elif (bmi > 18.5 and bmi < 24.9):
        return 'healthy'
    elif (bmi > 24.9 and bmi < 29.9):
        return 'overweight'
    else:
        return 'obese'


# In[21]:


data['health'] = data[['bmi']].apply(Health, axis= 1)


# In[22]:


data.head()


# In[23]:


sns.distplot(data['age'], color='darkred', bins=30)


# In[24]:


sns.distplot(data['avg_glucose_level'], color='darkred', bins=30)


# In[25]:


sns.distplot(data['bmi'], color='darkred', bins=30)


# In[26]:


sns.distplot(np.cbrt(data['age']), color='darkred', bins=30)


# In[27]:


sns.distplot(np.log(data['avg_glucose_level']), color='darkred', bins=30)


# In[28]:


sns.distplot(np.log(data['bmi']), color='darkred', bins=30)


# In[29]:


data['bmi_log'] = np.log(data['bmi'])
data['age_cbrt'] = np.sqrt(data['age'])


# In[30]:


data.columns


# In[31]:


dataset1 = data


# In[32]:


residence = pd.get_dummies(dataset1['Residence_type'], drop_first= True)
ever_married = pd.get_dummies(dataset1['ever_married'], drop_first= True)
gender = pd.get_dummies(dataset1['gender'], drop_first= True)
smoking_status = pd.get_dummies(dataset1['smoking_status'], drop_first= True)
work_type = pd.get_dummies(dataset1['work_type'], drop_first= True)
#health = pd.get_dummies(dataset1['health'], drop_first= True)


# In[33]:


dataset1.drop(['Residence_type', 'ever_married', 'gender', 'smoking_status', 'work_type', 'health'],
             axis= 1, inplace= True)


# In[34]:


dataset1 = pd.concat([dataset1,residence,ever_married,gender,smoking_status,work_type], axis= 1)


# In[35]:


dataset1.shape


# In[36]:


#Divide into test and train:
train1 = dataset1.loc[dataset1['source']=="train"]
test1 = dataset1.loc[dataset1['source']=="test"]

#Drop unnecessary columns:
test1.drop(['stroke','source'],axis=1,inplace=True)
train1.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train1.to_csv("train_modified1.csv",index=False)
test1.to_csv("test_modified1.csv",index=False)


# ## Model Building

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


# In[38]:


X_train,X_test,y_train,y_test= train_test_split(train1.drop(['stroke','bmi','age','avg_glucose_level','id'],axis=1), 
                                                    train1['stroke'], test_size=0.30, random_state=101)


# ### Logistic Regression Classifier

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


lrmod = LogisticRegression()
lrmod.fit(X_train, y_train)


# In[41]:


print (roc_auc_score(y_test, lrmod.predict_proba(X_test)[:,1]))

sol = test1.drop(['bmi','age','avg_glucose_level','id'], axis= 1)sol_prob = lgmod2.predict_proba(sol)submission = pd.DataFrame(columns=['id', 'stroke'])
submission['id'] = test1['id']
submission['stroke'] = sol_prob[:,1]submission.to_csv('Solution1.csv', index=False)
# ### XGBoost Classifier

# In[42]:


from xgboost import XGBClassifier


# In[43]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_prob = xgb.predict_proba(X_test)


# In[44]:


print (roc_auc_score(y_test, y_prob[:,1]))

sol_prob2 = xgb.predict_proba(sol)submission = pd.DataFrame(columns=['id', 'stroke'])
submission['id'] = test1['id']
submission['stroke'] = sol_prob2[:,1]submission.to_csv('Solution2.csv', index=False)