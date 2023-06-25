#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import calendar

# Get a list of all the month names
month_names = list(calendar.month_name)


# In[2]:


month_names[1:]


# # Help function

# In[3]:


def extract_date(text):
    pattern = r'\(([^)]*)\)[^()]*$'
    matches = re.findall(pattern, text)
    if len(matches):
        return matches[0][:4]
    else:
        return np.nan


# # Load data

# In[4]:


movies_df = pd.read_csv("MovieLenDataset/movies.csv")
ratings_df = pd.read_csv("MovieLenDataset/ratings.csv")
all_data = pd.merge(movies_df, ratings_df, on="movieId")


# # Get sorted dataframe for 'userId'

# In[5]:


# get one hot encoding for genres and movieId
items_our_user_rated = (all_data[all_data.userId==86].movieId).unique().tolist()
items_our_user_can_rate = all_data[~all_data.movieId.isin(items_our_user_rated)]["movieId"].unique().tolist()
all_moves_data = movies_df[movies_df.movieId.isin(all_data["movieId"].unique().tolist())]
all_moves_data = all_moves_data.filter(items=["movieId","genres"])
all_moves_data["genres"] = (all_moves_data["genres"].apply(lambda x : str(x).split(sep='|'))).values
dummies_genres = pd.get_dummies(all_moves_data['genres'].apply(pd.Series).stack()).groupby(level=0).sum()
dummies_movie = pd.get_dummies(all_moves_data["movieId"],prefix="movieId")
all_moves_data = pd.concat([all_moves_data,dummies_genres, dummies_movie],axis=1)
all_moves_data = all_moves_data[all_moves_data.movieId.isin(items_our_user_can_rate)]
all_moves_data = all_moves_data.drop(columns=["genres","movieId"])
   


# In[6]:


all_moves_data.iloc[:,:20]


# In[7]:


def prepare_data(userID):
   
    # get id for movies user can rate
    items_our_user_rated = (all_data[all_data.userId==userID].movieId).unique().tolist()
    items_our_user_can_rate = all_data[~all_data.movieId.isin(items_our_user_rated)]["movieId"].unique().tolist()
    user_dataFrame = pd.DataFrame()
   

    # get transaction time from movie rate
    transaction_from_movie_year = datetime.now().year - pd.to_datetime(movies_df[movies_df["movieId"].isin(items_our_user_can_rate)]["title"].apply(extract_date)).dt.year.values
    user_dataFrame["transaction_from_movie_year"] = transaction_from_movie_year


    # get three lag rate
    three_lag_rate = (all_data[all_data["userId"]==userID].sort_values("timestamp")).tail(3)["rating"].values
    user_dataFrame["lag_rate1"] = len(items_our_user_can_rate) * [three_lag_rate[2]]
    user_dataFrame["lag_rate2"] = len(items_our_user_can_rate) * [three_lag_rate[1]]
    user_dataFrame["lag_rate3"] = len(items_our_user_can_rate) * [three_lag_rate[0]]
  

    # get one hot encoding for genres and movieId
    all_moves_data = movies_df[movies_df.movieId.isin(all_data["movieId"].unique().tolist())]
    all_moves_data = all_moves_data.filter(items=["movieId","genres"])
    all_moves_data["genres"] = (all_moves_data["genres"].apply(lambda x : str(x).split(sep='|'))).values
    dummies_genres = pd.get_dummies(all_moves_data['genres'].apply(pd.Series).stack()).groupby(level=0).sum()
    dummies_movie = pd.get_dummies(all_moves_data["movieId"],prefix="movieId")
    all_moves_data = pd.concat([all_moves_data,dummies_genres, dummies_movie],axis=1)
    all_moves_data = all_moves_data[all_moves_data.movieId.isin(items_our_user_can_rate)]
    all_moves_data = all_moves_data.drop(columns=["genres","movieId"])
    

    # get one hot encoding for user id 
    col, row = len(all_data["userId"].unique()), len(items_our_user_can_rate)
    data = np.zeros((row,col))
    all_user_data = pd.DataFrame(data=data, columns=["userId_" + str(x) for x in list(all_data["userId"].unique())])
    all_user_data["userId_" +str(userID)] = 1
    
    # get one hot encoding for month
    col, row = 12, len(items_our_user_can_rate), 
    data = np.zeros((row,col)).astype(str)
    all_month_data = pd.DataFrame(data=data, columns=["transaction_month_"+str(x) for x in month_names[1:]])
    all_month_data["transaction_month_" + datetime.now().strftime("%B")] = 1

    # concate all data
    final_data = pd.concat([user_dataFrame.reset_index(),all_moves_data.iloc[:,:20].reset_index(),all_month_data.reset_index(),all_user_data.reset_index(),all_moves_data.iloc[:,20:].reset_index()],axis=1)
    
    return (final_data.drop(columns=["index"])).astype("float64")


# In[8]:


prepare_data(86)


# In[9]:


def predict(userID):
    items_our_user_rated = (all_data[all_data.userId==userID].movieId).unique().tolist()
    items_our_user_can_rate = all_data[~all_data.movieId.isin(items_our_user_rated)]["movieId"].unique().tolist()
    
    # prepare data
    user_data = prepare_data(userID)
    
    # predict from model
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    rate_pred = model.predict(np.array(user_data))
    
    # construct predicted dataframe
    predict_dataframe = pd.DataFrame()
    predict_dataframe["movieId"] = items_our_user_can_rate
    predict_dataframe["rates"] = rate_pred
    
    return predict_dataframe.sort_values("rates",ascending=False)
    


# In[ ]:





# In[10]:




# In[11]:




# In[ ]:




