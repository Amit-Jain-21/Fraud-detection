#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[ ]:





# In[7]:


test_id=pd.read_csv("/content/drive/MyDrive/Data/test_identity.csv")
test_txn=pd.read_csv("/content/drive/MyDrive/Data/test_transaction.csv")
train_id=pd.read_csv("/content/drive/MyDrive/Data/train_identity.csv")
train_txn=pd.read_csv("/content/drive/MyDrive/Data/train_transaction.csv")


# In[8]:


missing_values_count = train_id.isnull().sum()
print (missing_values_count[0:38])
total_cells = np.product(train_id.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


# In[9]:


from google.colab import drive
drive.mount('/content/drive')


# In[10]:


missing_values_count = train_txn.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_txn.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


# In[11]:


train = pd.merge(train_txn, train_id, on='TransactionID', how='left')
test = pd.merge(test_txn, test_id, on='TransactionID', how='left')


# In[12]:


train.head()


# In[13]:


test.head()


# In[14]:


print(train.isnull().any().sum())


# In[15]:


x=train['isFraud'].value_counts().values
sns.barplot([0,1],x)
plt.title('Target variable count')
plt.show()
print('  {:.2f}% of Transactions that are fraud in train '.format(train['isFraud'].mean() * 100))


# In[16]:


plt.hist(train['TransactionDT'], label='train');
plt.legend();
plt.title('Distribution of transactiond dates');


# In[17]:


plt.hist(test['TransactionDT'], label='test');
plt.legend();
plt.title('Distribution of transaction dates');


# In[18]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))
train.loc[train['isFraud'] == 1]     ['TransactionAmt'].apply(np.log)     .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Fraud',
          xlim=(-3, 10),
         ax= ax1)
train.loc[train['isFraud'] == 0]     ['TransactionAmt'].apply(np.log)     .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Not Fraud',
          xlim=(-3, 10),
         ax=ax2)
train.loc[train['isFraud'] == 1]     ['TransactionAmt']     .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Fraud',
         ax= ax3)
train.loc[train['isFraud'] == 0]     ['TransactionAmt']     .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Not Fraud',
         ax=ax4)
plt.show()


# Fraudulent charges appear to have a higher average transaction ammount

# In[19]:


print('Mean transaction amt for fraud is {:.2f}'.format(train.loc[train['isFraud'] == 1]['TransactionAmt'].mean()))
print('Mean transaction amt for non-fraud is {:.2f}'.format(train.loc[train['isFraud'] == 0]['TransactionAmt'].mean()))


# # Categorical Features - Identity
# We are told in the data description that the following Identity columns are categorical:
# 
# ###### DeviceType
# ###### DeviceInfo
# ###### id_12 - id_38
# 

# In[20]:


train.groupby('DeviceType')     .mean()['isFraud']     .sort_values()     .plot(kind='barh',
          figsize=(15, 5),
          title='Percentage of Fraud by Device Type')
plt.show()


# In[21]:


train.groupby('DeviceInfo')     .count()['TransactionID']     .sort_values(ascending=False)     .head(20)     .plot(kind='barh', figsize=(15, 5), title='Top 20 Devices in Train')
plt.show()


# # Categorical Features - Transaction
# We are told in the data description that the following transaction columns are categorical:
# 
# ##### ProductCD 
# ##### emaildomain 
# ##### card1 - card6 
# ##### addr1, addr2 
# ##### P_emaildomain 
# ##### R_emaildomain 
# ##### M1 - M9 
# ## card1 - card6
# We are told these are all categorical, even though some appear numeric.

# In[22]:


card_cols = [c for c in train.columns if 'card' in c]
train[card_cols].head()


# In[23]:


color_idx = 0
for c in card_cols:
    if train[c].dtype in ['float64','int64']:
        train[c].plot(kind='hist',
                                      title=c,
                                      bins=50,
                                      figsize=(15, 2),
                                      color=color_pal[color_idx])
    color_idx += 1
    plt.show()
  


# In[24]:


train_fr = train.loc[train['isFraud'] == 1]
train_nofr = train.loc[train['isFraud'] == 0]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_fr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax1, title='Count of card4 fraud')
train_nofr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax2, title='Count of card4 non-fraud')
train_fr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax3, title='Count of card6 fraud')
train_nofr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax4, title='Count of card6 non-fraud')
plt.show()


# In[ ]:




