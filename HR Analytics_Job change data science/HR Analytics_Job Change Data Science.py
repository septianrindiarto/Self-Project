#!/usr/bin/env python
# coding: utf-8

# # HR Analytics: Job Change of Data Science

# **Features**
# 
# - enrollee_id : Unique ID for candidate
# - city: City code
# - city_ development _index : Developement index of the city (scaled)
# - gender: Gender of candidate
# - relevent_experience: Relevant experience of candidate
# - enrolled_university: Type of University course enrolled if any
# - education_level: Education level of candidate
# - major_discipline :Education major discipline of candidate
# - experience: Candidate total experience in years
# - company_size: No of employees in current employer's company
# - company_type : Type of current employer
# - lastnewjob: Difference in years between previous job and current job
# - training_hours: training hours completed
# - target: 0 – Not looking for job change, 1 – Looking for a job change
# 
# 
# **Task**
# 
# - Predict the probability of a candidate will work for the company
# - Interpret model(s) such a way that illustrate which features affect candidate decision
# - HR Analytics

# # Importing Tools and Data

# In[143]:


# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, roc_auc_score, f1_score, log_loss, matthews_corrcoef, log_loss


# In[2]:


data_info = pd.read_excel('Feature Definition.xlsx', header=1, index_col='Feature')

def feat_info(col_name):
    print(data_info.loc[col_name]['Definition'])


# In[3]:


train = pd.read_csv('aug_train.csv')
test = pd.read_csv('aug_test.csv')


# # Data Overview

# In[4]:


train


# In[5]:


test


# In[6]:


train.info()


# In[7]:


train.select_dtypes(include='object').describe().transpose()


# In[8]:


listItem = []

for col in train.columns:
    listItem.append([col, train[col].dtype, train[col].isna().sum(), round((train[col].isna().sum())/len(train[col])*100), 
                    train[col].nunique(), list(train[col].drop_duplicates().sample(2).values)])

dfDesc = pd.DataFrame(columns=['data features', 'dataType', 'null', 'nullPct%', 'unique', 'uniqueSample'], data=listItem)
dfDesc


# Now that we know about percentage of null values each feature, let's get over it and then let's make encoding of the categorical feature so we can visualize the data and determine wheter the feature is useful for modelling or not.

# # Preprocessing Data

# In[9]:


# let's drop the values which the percetage of null values below
train.dropna(subset=['enrolled_university', 'education_level', 'last_new_job', 'experience'], inplace=True)


# In[10]:


train.isna().sum()


# In[11]:


train.info()


# Let's focus on determining whether change the gender, relevant_experience, major_dicipline, company_size, and company_type value or drop its null values.

# In[12]:


feat_info('gender')
feat_info('relevant_experience')
feat_info('major_dicipline')
feat_info('company_size')
feat_info('company_type')


# We know that except company size, we cannot fill the null value because we know we have to put a concrete essence for the value of those feature. So now we make change the null value with others

# In[13]:


train['gender'].unique()


# In[14]:


train['major_discipline'].unique()


# In[15]:


train['company_type'].unique()


# From the information above we can change the null values into "Other", so that we not assuming null value with misunderstanding. Let's change the null values of those feature into "Other".

# In[16]:


train[['gender', 'major_discipline', 'company_type']] = train[['gender', 'major_discipline', 'company_type']].fillna('Other')
print(train['gender'].unique())
print(train['major_discipline'].unique())
print(train['company_type'].unique())


# In[17]:


train.info()


# # Explanatory Data Analysist

# In[18]:


train = train.drop('enrollee_id', axis=1)


# In[19]:


sns.countplot(train['target'])


# Looks like we got an imbalance target. Later we should choose wether we balance it or not.

# In[20]:


sns.pairplot(train)


# In[21]:


sns.heatmap(train.corr(), annot=True)


# From graphic above we could tell that there are insignificant correlation between city development index and training hours to the target. Let's take a look the distribution of each city development index and training hours to the target

# In[22]:


sns.lineplot(x='target', y='training_hours', data=train, color='blue')


# In[23]:


sns.lineplot(x='target', y='city_development_index', data=train, color='blue')


# It seems that both of the feature has the same pattern which is if value is getting smaller then they tend to looking for job change.

# Now let's find out whether if they had more education level they tend to change the job or not and how many of them has a relevant experience. Hereby we want to know that the habbit of someone with relevant experience to change their job. From that point we could know that Data Science is a preferable job or not to those who already had experience with relevant job.

# In[24]:


train.select_dtypes(include='object').keys()


# In[25]:


sns.countplot(train['relevent_experience'], hue=train['target'])
# bikin secara persentasinya


# In[26]:


train[train['relevent_experience']=='No relevent experience']['target'].value_counts()[1]/len(train[train['relevent_experience']=='No relevent experience']['target'])


# In[27]:


sns.countplot(train['education_level'], hue=train['target'])


# In[28]:


order = ['<1','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','>20']

plt.figure(figsize=(10,5))
sns.countplot(train['experience'], hue=train['target'], order=order)


# We can conclude that people who tend to change job has no relevent experience with education equal to graduate and most of them likely has experience to work at least 4 years.

# # Training Data

# We still have null values on company size. Let's fill it with KNN imputer. Let's take a look at the feature.

# In[29]:


feat_info('company_size')


# In[30]:


train['company_size'].unique()


# Looks like the data itself is clustered situation. Let's fill the null value with KNN model.

# In[31]:


from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')


# In[32]:


train.info()


# In[33]:


obj = train.select_dtypes(include='object').astype('category')
flt = train.select_dtypes(exclude='object').reset_index()


# In[34]:


cat = obj.apply(lambda x: x.cat.codes)


# In[35]:


imputer.fit(cat)


# In[36]:


cat_trans = imputer.transform(cat)


# In[37]:


fix = pd.DataFrame(cat_trans, columns=obj.columns)


# In[38]:


fix.info()


# In[39]:


df = pd.concat([fix,flt], axis=1)
df.info()


# In[40]:


X = df.drop('target', axis=1)
y = df['target']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)


# # Creating Model

# In[42]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[43]:


def lg_predict(X_train, y_train):
    # calling the model
    model = LogisticRegression()
    
    # fit the data
    model.fit(X_train, y_train)
    
    # predicting the target
    predict = model.predict(X_test)
    
    # making confusion matrix and classification report
    print('Logistic Regression Report')
    print(classification_report(y_test, predict))
    print(confusion_matrix(y_test, predict))

def svc_predict(X_train, y_train):
    # calling the model
    model = SVC()
    
    # fit the data
    model.fit(X_train, y_train)
    
    # predicting the target
    predict = model.predict(X_test)
    
    # making confusion matrix and classification report
    print('SVC Report')
    print(classification_report(y_test, predict))
    print(confusion_matrix(y_test, predict))
    
def knn_predict(X_train, y_train):
    # calling the model
    model = KNeighborsClassifier()
    
    # fit the data
    model.fit(X_train, y_train)
    
    # predicting the target
    predict = model.predict(X_test)
    
    # making confusion matrix and classification report
    print('KNN Report')
    print(classification_report(y_test, predict))
    print(confusion_matrix(y_test, predict))
    
def rfc_predict(X_train, y_train):
    # calling the model
    model = RandomForestClassifier()
    
    # fit the data
    model.fit(X_train, y_train)
    
    # predicting the target
    predict = model.predict(X_test)
    
    # making confusion matrix and classification report
    print('Random Forest Report')
    print(classification_report(y_test, predict))
    print(confusion_matrix(y_test, predict))


# In[44]:


lst = [lg_predict, svc_predict, knn_predict, rfc_predict]

for i in lst:
    print
    i(X_train, y_train)


# From evaluation metrics above we can see that random forest has a better result than the others so we continue tuning the paramater of random forest in search for a better result.

# # Hyperparameter Tuning

# In[45]:


params_rf = [{
    'n_estimators' : [1, 5, 10, 50, 100,500, 1000],
    'max_features' : ['auto', 'sqrt'],
    'max_depth' : [2, 4, 6, 8, 10],
    'min_samples_split' : [2,5],
    'min_samples_leaf' : [1,2],
    'bootstrap' : [True, False]
}]


# In[46]:


cv = GridSearchCV(RandomForestClassifier(), param_grid=params_rf, cv=3, verbose=True, n_jobs=-1)


# In[47]:


cv_tuned = cv.fit(X_train, y_train)


# In[54]:


cv_tuned.best_params_


# In[48]:


predict = cv_tuned.predict(X_test)
print(classification_report(y_test, predict))
print(confusion_matrix(y_test, predict))


# # Boosting Evaluation Metrics

# In[49]:


# try using auc roc and display its visualization
# evaluate it using kfold
# gudluck


# In[61]:


from sklearn.model_selection import KFold
K = 100
kf =KFold(n_splits = K, shuffle = True, random_state = 42)


# In[62]:


def calc_train_error(X_train, y_train, model):
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    matt = matthews_corrcoef(y_train, predictions)
    f1 = f1_score(y_train, predictions, average ='macro')
    report = classification_report(y_train, predictions)
    roc_auc = roc_auc_score(y_train, predictProba[:, 1])
    accuracy = accuracy_score(y_train, predictions)
    confMatrix = confusion_matrix(y_train, predictions)
    logloss = log_loss(y_train,predictProba)
    return{
        'report' : report, 
        'matthew' : matt,
        'f1' : f1,
        'roc': roc_auc,
        'accuracy': accuracy,
        'confusion': confMatrix,
        'logloss' : logloss
    }

def calc_validation_error(X_test, y_test, model):
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    matt = matthews_corrcoef(y_test, predictions)
    f1 = f1_score(y_test, predictions, average ='macro')
    report = classification_report(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictProba[:, 1])
    accuracy = accuracy_score(y_test, predictions)
    confMatrix = confusion_matrix(y_test, predictions)
    logloss = log_loss(y_test,predictProba)
    return{
        'report' : report, 
        'matthew' : matt,
        'f1' : f1,
        'roc': roc_auc,
        'accuracy': accuracy,
        'confusion': confMatrix,
        'logloss' : logloss
    }

def calc_metrics(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

def visualize_error()


# In[63]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(X, y):
    
    #split data
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    
    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, RandomForestClassifier(
        bootstrap=True, max_depth=10, max_features='auto', min_samples_leaf=2, min_samples_split=5, n_estimators=100))
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[172]:


lst = []
for i, tr_err, val_err in zip(range(1,100,1), train_errors, validation_errors):
    error = tr_err['roc']-val_err['roc']
    lst.append(error)
lst = pd.DataFrame(lst)
best_iteration_score = lst.reset_index().rename(columns={'index':'iteration', 0:'error_score'}).sort_values(by='error_score')
best_iteration_score.iloc[0]


# In[173]:


print('Best iteration is at {} and the error score is {}'.format(best_iteration_score['iteration'].iloc[0],
                                                                 best_iteration_score['error_score'].iloc[0]))


# In[174]:


best_iteration_score.iloc[-1]


# Now that we know about the best iteration and the minimun error, let's find out the classification report according to the 65th iteration.

# In[166]:


train_error, test_error = [], []
for i, tr_err, val_err in zip(range(1,100,1), train_errors, validation_errors):
    train_error.append(tr_err['roc'])
    test_error.append(val_err['roc'])
    
plt.figure(figsize = (15,8))
plt.plot(train_error, 'r-+', label = 'Training Loss')
plt.plot(test_error, 'b-', label = 'Test Loss')
plt.xlabel('Number Of Batches')
plt.ylabel('Log-Loss')
plt.legend(loc = 'best')

plt.show()


# In[161]:


# funtion to plot learning curves

def plot_learning_curve(model, X, Y):
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 11)
    train_loss, test_loss = [], []
    
    for m in range(200,len(x_train),200):
        
        model.fit(x_train.iloc[:m,:], y_train[:m])
        y_train_prob_pred = model.predict_proba(x_train.iloc[:m,:])
        train_loss.append(log_loss(y_train[:m], y_train_prob_pred))
        
        y_test_prob_pred = model.predict_proba(x_test)
        test_loss.append(log_loss(y_test, y_test_prob_pred))
        
    plt.figure(figsize = (15,8))
    plt.plot(train_loss, 'r-+', label = 'Training Loss')
    plt.plot(test_loss, 'b-', label = 'Test Loss')
    plt.xlabel('Number Of Batches')
    plt.ylabel('Log-Loss')
    plt.legend(loc = 'best')



    plt.show()
        


# In[162]:


plot_learning_curve(RandomForestClassifier(
        bootstrap=True, max_depth=10, max_features='auto', min_samples_leaf=2, min_samples_split=5, n_estimators=100),
                    X, y)


# Now that we know the gap between the train and test gap of loss. But we can still boost it cause we know that target has an imbalance data so perhaps we can find better evaluation if we make it balance using SMOTE.

# In[167]:


from imblearn.over_sampling import SMOTE


# In[169]:


smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(X,y)


sns.countplot(Y_smote, edgecolor = 'black')


# In[171]:


plot_learning_curve(RandomForestClassifier(
        bootstrap=True, max_depth=10, max_features='auto', min_samples_leaf=2, min_samples_split=5, n_estimators=100),
                    X_smote, Y_smote)


# In[ ]:




