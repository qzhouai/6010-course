#!/usr/bin/env python
# coding: utf-8

# In[34]:


## the data contains expiration date and the registration init time. we convert this into membership days.
## this script is based on the TalySacc's Script. I  understand and analyze the code, made a bit revise.
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.metrics  import log_loss


# In[35]:


import lightgbm as lgb


# In[36]:


INPUT_DATA_PATH  = '/Users/zhouquan/Desktop/'  # this is to help get the path of the input data, different computer
# have different path
df_train = pd.read_csv(INPUT_DATA_PATH + 'train.csv', dtype = {'msno' : 'category', 'source_system_tab':'category',
                                                              'source_screen_name':'category', 'source_type':'category', 
                                                              'target': np.uint8,  'song_id' : 'category'})
## set the datatype


# In[37]:


df_train  # observe the data 


# In[38]:


df_test  = pd.read_csv(INPUT_DATA_PATH + 'test.csv', dtype = {'msno':'category', 'source_system_tab':'category', 
                                                             'source_screen_name': 'category', 'source_type': 'category', 
                                                             'song_id': 'category'} )
## set the datatype 


# In[39]:


df_test


# In[40]:


df_members = pd.read_csv(INPUT_DATA_PATH + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'},
                                                      parse_dates=['registration_init_time','expiration_date'])


# In[41]:


df_members


# In[42]:


# we have import the data successfully and set the datatype.
# use the information, find that we can transfer the start date and end date to membership days.
df_members['membership_days'] = (df_members['expiration_date']  - 
                                 df_members['registration_init_time']).dt.days.astype(int)
## delete two lines: registration_init time and expiration_date, since we already used the information
df_members = df_members.drop(['registration_init_time','expiration_date'], axis=1)


# In[43]:


df_members


# In[44]:


df_test  =  pd.merge (left = df_test,right = df_members,how='left',on='msno')
## merger the df_test and the df_members
## msno columns are the same, put df_test in the left, df_members on the right. Aggregate the information
df_test.msno = df_test.msno.astype('category')


# In[45]:


## merge the member dataframe into df_train dataframe and the songs dataframe into df_test, df_train dataframes 
df_train = pd.merge(left = df_train,right = df_members,how='left',on='msno')
df_train.msno = df_train.msno.astype('category')

df_songs = pd.read_csv(INPUT_DATA_PATH+ 'songs.csv', dtype = {'genre_ids' : 'category', 'artist_name':'category',
                                                             'language':'category', 'composer':'category', 'lyricist':'category', 
                                                             'song_id':'category'}) 


# In[46]:


# merge the songs dataframe into test, train(which has been merged before)
df_test = pd.merge(left = df_test,right = df_songs,how = 'left',on='song_id')
df_train = pd.merge(left = df_train,right = df_songs,how = 'left',on='song_id')


# In[47]:


# we deal with the song length column that are na terms and refill them with 300000, set the type of the data.
df_test.song_length.fillna(300000,inplace=True)
df_test.song_length = df_test.song_length.astype(np.uint32)
df_test.song_id = df_test.song_id.astype('category')
# same rule with the training data processing
df_train.song_length.fillna(300000,inplace=True)
df_train.song_length = df_train.song_length.astype(np.uint32)
df_train.song_id = df_train.song_id.astype('category')


# In[50]:


import lightgbm as lgb

# Create a Cross Validation with 5 splits
kf = KFold(n_splits=5)

# This will store the predictions.
predictions = np.zeros(shape=[len(df_test)])

# For each KFold, we will get:
for train_indices ,validate_indices in kf.split(df_train) : 
   train_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[train_indices,:],label=df_train.loc[train_indices,'target'])
   val_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[validate_indices,:],label=df_train.loc[validate_indices,'target'])
   # Create parameters for LGBM
   params = {
       'objective': 'binary',
       'metric': 'binary_logloss',
       'boosting': 'gbdt',
       'learning_rate': 0.2 ,
       'verbose': 0,
       'num_leaves': 100,
       'bagging_fraction': 0.95,
       'bagging_freq': 1,
       'bagging_seed': 1,
       'feature_fraction': 0.9,
       'feature_fraction_seed': 1,
       'max_bin': 128,
       'max_depth': 10,
       'num_rounds': 200,
       'metric' : 'auc',
       } 
    # Train the model
   bst = lgb.train(params, train_data, 100, valid_sets=[val_data])
   
   # Make the predictions and save them on the predictions array which has been set before 
   predictions += bst.predict(df_test.drop(['id'],axis=1))
   
   # Release the model from memory for the next iteration
   del bst

print('Training process finished. Generating Output...')

# We get the amount of 5 predictions from the prediction list, and divide the predictions by 5, which equals to numbers
# of the kfolds
predictions = predictions/5

# Read the sample_submission CSV
submission = pd.read_csv(INPUT_DATA_PATH + '/sample_submission.csv')
# Set the target to our predictions
submission.target=predictions
# Save the submission file
submission.to_csv('submission.csv',index=False)
print('Output created.')


# In[51]:





# In[ ]:




