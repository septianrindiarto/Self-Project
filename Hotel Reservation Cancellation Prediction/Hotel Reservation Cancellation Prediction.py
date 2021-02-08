#!/usr/bin/env python
# coding: utf-8

# # Hotel Reservation Cancellation Prediction

# Feature definition can be accessed at
# https://www.sciencedirect.com/science/article/pii/S2352340918315191

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, f1_score, log_loss, matthews_corrcoef
from scipy.stats import randint


# ## Data Cleaning & Preprocessing

# In[2]:


# importing Dataset
df = pd.read_csv('hotel_bookings.csv')


# In[3]:


# membatasi dataset menjadi 5000 data dan variabel yang digunakan
df = df[['hotel', 'is_canceled', 'adults', 'children', 'babies',
'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'booking_changes',
'deposit_type', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
'total_of_special_requests']].head(5000)


# In[4]:


# melihat overview dataset secara keseluruhan
df


# In[5]:


# melihat jumlah kolom/variabel yang diperlukan
df.info()


# Dari data tersebut dapat kita lihat bahwa ada 8 type object, 7 type integer dan 1 type float. Namun perlu diperhatikan bahwa tidak semua tipe data integer ataupun float bersifat categorical. Oleh karena itu, kita perlu membedah lebih dalam mana kolom yang termasuk kategorikal atau tidak.

# In[6]:


dfDesc = []

for i in df.columns:
        dfDesc.append([i, df[i].dtypes, df[i].isna().sum(), round((((df[i].isna().sum())/(len(df)))*100),2), 
                    df[i].nunique(), df[i].drop_duplicates().sample(1).values])
pd.DataFrame(dfDesc, columns = ['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'])


# Data tersebut menunjukkan bahwa dataset yang kita miliki semua bersifat categorical. Selanjutnya kita akan fokus menangani data null pada kolom country.

# In[7]:


df[df.isna().any(axis=1)]


# Data tersebut menunjukkan adanya row kosong pada dataset sehingga tidak perlu kita gunakan untuk proses lebih lanjut.

# In[8]:


df.dropna()


# In[9]:


# melihat jumlah data Null kembali setelah penghapusan data Null
df.info()


# Data sudah bersih dari Null dan siap untuk digunakan. Data Null digantikan dengan data lain yang diambil dari total dataset.

# ## EDA

# Hal yang perlu diperhatikan adalah:
#     1. Memahami profil tamu/konsumen hotel (customer profiling).
#     2. Memahami kebiasaan tamu/konsumen hotel (customer behavior).
#     3. Analisa perbedaan ciri-ciri transaksi booking yang berhasil dan cancel.

# untuk memahami profil konsumen kita perlu lebih fokus melihat sebaran data adults, children, babies, mmarket segment, required car parking spaces.

# In[10]:


# mempersempit view agar mudah untuk di analisis
df_profile = df[['adults', 'children', 'babies', 'market_segment', 'required_car_parking_spaces']]


# In[11]:


df_profile


# In[12]:


for i in list(df_profile.columns):
    locs, labels = plt.xticks()
    a = sns.countplot(x=i, data=df)
    plt.setp(labels, rotation=40)
    plt.show()


# Data tersebut menggambarkan bahwa:
# 1. Banyaknya tamu jumlah orang dewasa per kamar = 2
# 2. Mayoritas tamu tidak membawa anak
# 3. Mayoritas tamu tidak membawa bayi
# 4. Mayortias tamu berasal dari Travel Agent
# 5. Mayoritas tamu tidak membutuhkan tempat parkir mobil
# Keterangan tersebut menggambarkan karakter tamu yang berkunjung.
# 
# Dari data tersebut pun dapat kita ambil keterengan bahwa kebiasaan tamu yang adalah mereka kemungkinan pasangan yang sedang bulang madu atau dalam perjalanan dinas mengingat jarangnya tamu yang membawa keluarga, tidak memerlukan parkir mobil dan kebanyakan dari mereka membeli tiket melalui travel agensi.

# In[13]:


for i in list(df.drop('is_canceled', axis=1).columns):
    locs, labels = plt.xticks()
    a = sns.countplot(x=i, hue='is_canceled', data=df)
    plt.setp(labels, rotation=40)
    plt.show()


# Data Tersebut pada dasarnya tidak menjelaskan ciri-ciri yang signifikan menunjukkan adanya variabel yang kuat mempengaruhi tingkat cancel booking tamu. Namun secara garis besar yang perlu kita ketahui adalah tipe customer yang hanya sementara.

# ## Model Building & Hyper-parameter Tuning

# In[14]:


# menentukan variabel dependent dan independet
target = df['is_canceled']
data = df.drop('is_canceled', axis=1)


# Dalam kasus ini kami akan menggunakan 3 supervised model yaitu Decision Tree, Random Forest, dan XGBoost

# Sebelum melakukan pemodelan terlebih dahulu kita menganalisis target dan melakukan train test split

# ### Train Test Split

# In[15]:


target.to_frame()


# In[16]:


sns.countplot(target)


# Target memiliki jumlah yang balance sehingga tidak perlu dilakukan scaler

# In[17]:


data


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, roc_curve,accuracy_score, matthews_corrcoef, auc, log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[19]:


# melakukan one hot encoding via get_dummies
data = pd.get_dummies(data, drop_first=True)
target = pd.get_dummies(target, drop_first=True)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(
    data, 
    target, 
    random_state=101, test_size=0.3)


# In[21]:


# melihat overview data hasil encoding
y_test


# ### Decision Tree

# In[22]:


# assign parameter
params_dtree = [{
    'max_depth' : [1,2,3,4,5,6,7,8,9,10,None],
    'criterion' : ['gini', 'entropy']
}]


# In[23]:


dtree = DecisionTreeClassifier()

cv = GridSearchCV(dtree, 
                  param_grid=params_dtree, cv=3, verbose=True, n_jobs=-1)


# In[24]:


cv.fit(X_train, y_train)


# In[25]:


cv.best_params_


# In[26]:


dtree = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10)


# In[27]:


dtree.fit(X_train, y_train)


# In[ ]:





# ### Random Forest

# In[28]:


# assign parameter
params_rf = [{
    'n_estimators' : [1, 5, 10, 50, 100],
    'max_features' : ['auto', 'sqrt'],
    'max_depth' : [2, 4, 6, 8, 10],
    'min_samples_split' : [2,5],
    'min_samples_leaf' : [1,2],
    'bootstrap' : [True, False]
}]

rfc = RandomForestClassifier()


# In[29]:


cv = GridSearchCV(rfc, param_grid=params_rf, cv=3, verbose=True, n_jobs=-1)

cv.fit(X_train, y_train)


# In[30]:


cv.best_params_


# In[ ]:


rfc = RandomForestClassifier(bootstrap=False, 
                             max_depth=10, 
                             max_features='auto', 
                             min_samples_leaf=1,  
                             min_samples_split=5, n_estimators=100)


# In[ ]:


rfc.fit(X_train, y_train)


# ### XGBoost

# In[33]:


params_xgboost = [{
    'learning_rate'    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'max_depth'        : [3, 4, 5, 6, 8, 10, 12, 15],
    'gamma'            : [1, 3, 5, 7],
    'colsample_bytree' : [0.3, 0.4, 0.5, 0.7]
}]

xgb = XGBClassifier()


# In[34]:


cv = GridSearchCV(xgb, param_grid=params_xgboost, cv=3, verbose=True, n_jobs=-1)

cv.fit(X_train, y_train)


# In[35]:


cv.best_params_


# In[36]:


xgb = XGBClassifier(colsample_bytree=0.4, 
                    gamma=3, learning_rate=0.2, max_depth=12)


# In[37]:


xgb.fit(X_train, y_train)


# Masing-masing model telah dibuat dan ditetapkan parameter-parameter khusus beserta range nya sehingga kita dapat mengetahui parameter terbaik yang dapat kita gunakan untuk menganalisis data.

# ## Model Evaluation

# Ada 4 point penting yang perlu diperhatikan, point tersebut:
# 
# 1. Pilih jenis kesalahan yang paling berpengaruh pada kerugian finansial perusahaan dan jelaskan alasan
# pilihan Anda!
# 2. Gunakan serta jelaskan output dari ROC AUC dan Classification Report (Precision, Recall, & Acuracy)!
# Setiap evaluasi model , gunakan metode K-Fold Cross Validation untuk semua metric-nya. Tampilkan
# seluruh hasil evaluation metric dalam bentuk DataFrame.
# 3. Jelaskan langkah Anda untuk meningkatkan performa model ML untuk bisa menekan jenis kesalahan
# tersebut!
# 4. Beri kesimpulan di akhir meliputi: 1) model ML yang dipilih, dan 2) impact ke bisnis!

# langkah pertama untuk menjabarkan point-point tersebut adalah dengan menjelaskan hasil classification report dan ROC AUC terlebih dahulu.

# Adapun penjelasan mengenai Classification report dan ROC AUC, lebih baik kita pahami defini dari Classification report dan ROC AUC

# 1. Accuracy: proporsi jumlah total prediksi yang benar. Saat kami menggunakannya: Saat kami memiliki target keseimbangan pada set data kereta dan pengujian. Contoh: ketika membangun model untuk mengklasifikasikan spam atau bukan spam dengan target seimbang di set data train dan test.
# 1. Recall: proporsi kasus positif aktual yang diidentifikasi dengan benar. Ketika kita menggunakannya: ketika kita memiliki target yang tidak seimbang pada set data kereta atau pengujian bersama dengan kita ingin menghitung semua kasus positif aktual masuk Contoh: ketika kita membangun model penipuan, kita tidak ingin penipuan yang sebenarnya tidak terdeteksi.
# 1. Precision: proporsi kasus positif yang diprediksi yang diidentifikasi dengan benar. Saat kami menggunakannya: ketika kami memiliki target yang tidak seimbang di kereta atau set data pengujian bersama dengan kami ingin menghitung semua kasus positif yang diprediksi masuk. Contoh: Saat kami membuat model rekomendasi video.
# 1. Kurva FPR-TPR (ROC): grafik yang menunjukkan kinerja model klasifikasi di semua ambang klasifikasi. AUC KOP: Area di bawah kurva KOP. Saat kami menggunakannya: ketika kami ingin mempertimbangkan semua kinerja model pada ambang mana pun bersama dengan kami ingin menghitung semua kasus positif aktual masuk Contoh: ketika kami membangun model penipuan, kami tidak ingin penipuan yang sebenarnya tidak terdeteksi.

# ### Langkah 1. Membandingkan hasil Classification Report

# In[38]:


#dtree
predict = dtree.predict(X_test)
print(classification_report(y_test, predict))


# In[39]:


#Random Forest
predict = rfc.predict(X_test)
print(classification_report(y_test, predict))


# In[40]:


#XGBoost
predict = xgb.predict(X_test)
print(classification_report(y_test, predict))


# Dalam kasus dataset ini kita akan fokus pada poin **Recall** karena perusahaan harus benar-benar poin aktual positif yang teridentifikasi benar. Dengan demikian prediksi tamu yang hendak booking secara aktual teridentifikasi benar.

# Ketiga Model diatas **Random Forest** lah yang memiliki nilai recall paling tinggi.

# ### Langkah 2. Melihat AUC ROC

# In[41]:


X_train, X_test, y_train, y_test = train_test_split(
    data, 
    target, 
    random_state=101, test_size=0.3)


# In[43]:


predictions_proba = xgb.predict_proba(X_test)


# In[44]:


predictions_proba


# In[45]:


sns.distplot(predictions_proba[:,0])


# In[46]:


np.sum(predictions_proba[:,0] < 0.5)/len(predictions_proba)


# In[62]:


dfn = pd.DataFrame()


# In[64]:


dfn['pred_score'] = predictions_proba[:,0]


# In[65]:


dfn


# In[61]:


df_tmp.drop('y_test', axis=1)
df_tmp


# In[69]:


for enum, i in enumerate(np.linspace(0,1,11)):
    dfn['y_pred_{}'.format(enum)] = dfn['pred_score'].apply(
        lambda x: 1 if x < i else 0)


# In[70]:


dfn


# In[71]:


for i in range(0,11):
    tp = len(df_tmp[(df_tmp['y_test'] == 1) & (df_tmp['y_pred_{}'.format(i)] == 1)])
    fn = len(df_tmp[(df_tmp['y_test'] == 1) & (df_tmp['y_pred_{}'.format(i)] == 0)])
    fp = len(df_tmp[(df_tmp['y_test'] == 0) & (df_tmp['y_pred_{}'.format(i)] == 1)])
    recall = tp/(fn + tp)
    errors = fn + fp
    print('recall treshold ke-{} adalah {} dengan errors {}'.format(i,recall,errors))


# Dengan melihat angka diatas kita memilih treshold ke-8 yaitu 1 karena menghasilkan recall yang paling baik dengan jumlah errors paling sedikit

# Dalam kasus ini, sejumlah nilai di FN (False Negative) akan mengakibatkan kerugian finansial aktual kepada perusahaan. Sedangkan sejumlah nilai di FP (False Positive) akan menghilangkan business opportunity kepada perusahaan. Jelas yang paling berpengaruh adalah merupakan kerugian aktual bagi perusahaan.

# In[72]:


from sklearn.model_selection import KFold
K = 5
kf =KFold(n_splits = K, shuffle = True, random_state = 42)


# In[73]:


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


# In[74]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, df['is_canceled']):
    
    #split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = df['is_canceled'].iloc[train_index], df['is_canceled'].iloc[val_index]
    
    
    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, xgb)
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[75]:


train_errors


# In[76]:


for i, tr_err, val_err in zip(range(1,6,1), train_errors, validation_errors):
    print('ROC AUC Train ke {} : {}'.format(i, tr_err['roc']))
    print('ROC AUC Validation ke {} : {}'.format(i, val_err['roc']))


# In[77]:


for i, tr_err, val_err in zip(range(1,6,1), train_errors, validation_errors):
    print('Accuracy Train ke {} : {}'.format(i, tr_err['accuracy']))
    print('Accuracy Validation ke {} : {}'.format(i, val_err['accuracy']))


# In[78]:


for i, tr_err, val_err in zip(range(1,6,1), train_errors, validation_errors):
    print(f'Report Train ke {i} :')
    print(tr_err['report'])
    print(f'Report Validation ke {i} :')
    print(val_err['report'])


# Hasil validasi terbaik berada pada percobaan ke 2 dan 4 dengan hasil recall sebesar 0.94

# # DONE
