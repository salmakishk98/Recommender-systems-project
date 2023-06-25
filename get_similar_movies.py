#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv("MovieLenDataset/movies.csv")

# In[4]:


ratings=pd.read_csv("MovieLenDataset/ratings.csv")





#


# In[7]:


utility_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)


# In[8]:


def adjusted_cosine_sim_vectorized(utility_matrix,movie_list):
    """
    Returns Construct item-item (Adjusted Cosine) similarity matrix .

            Parameters:
                utility matrix : Movies(item) as index and users as column --> shape(num_movies, number_users)

            Returns:
                similarity matrix
    """
    norm_utility_matrix = utility_matrix - np.mean(utility_matrix ,axis=1).reshape(utility_matrix.shape[0],-1)
    # calculate norm for all item vectors
    all_item_norm = np.linalg.norm(norm_utility_matrix ,axis=1).reshape(1, norm_utility_matrix.shape[0])
    similarity_matrix = (norm_utility_matrix @ norm_utility_matrix.T) / (all_item_norm.T * all_item_norm)
    similarity_matrix = pd.DataFrame(data=np.round(similarity_matrix,2), columns=movie_list, index=movie_list)
    return similarity_matrix


# In[9]:


def similar_items_by_id(movie_id):
    similarity_matrix = adjusted_cosine_sim_vectorized(np.array(utility_matrix),movie_list=list(utility_matrix.index))
    sorted=similarity_matrix[movie_id].sort_values(ascending=False)[1:]
    ids =sorted.index.tolist()
    filtered_movies = movies.loc[movies["movieId"].isin(ids)]
    sorted_df = filtered_movies.sort_values(by='movieId', key=lambda x: x.map({v: i for i, v in enumerate(ids)}))
    sorted_df["similarity score"]=sorted.values
    return sorted_df







links=pd.read_csv("MovieLenDataset/links.csv")
watch_count = ratings.groupby(['movieId']).agg({'userId': 'count'}).reset_index()
watch_count.columns=["movieId","watch count"]




available_ratings=ratings["movieId"].unique().tolist()


movies=movies[movies["movieId"].isin(available_ratings)]




watch_count = ratings.groupby(['movieId']).agg({'userId': 'count'}).reset_index()
watch_count.columns=["movieId","watch count"]
avg_rating=ratings.groupby("movieId").agg({"rating":"mean"}).reset_index()
avg_rating.columns=["movieId","avg ratings"]
movies_merged=pd.merge(movies,links[["tmdbId","imdbId","movieId"]], on=["movieId"]).merge(watch_count[["movieId","watch count"]],on=["movieId"])
movies_metadata=pd.merge(movies_merged,avg_rating[["movieId","avg ratings"]], on=["movieId"])



# In[21]:


def similar_item_by_name(movie_name):
    if movie_name not in movies['title'].values:
        return "no movie in database"
    movie_id = movies_metadata.loc[movies_metadata['title'] ==movie_name, 'movieId'].values[0]
    similarity_matrix = adjusted_cosine_sim_vectorized(np.array(utility_matrix),movie_list=list(utility_matrix.index))
    sorted=similarity_matrix[movie_id].sort_values(ascending=False)[1:]
    ids =sorted.index.tolist()
    filtered_movies = movies_metadata.loc[movies_metadata["movieId"].isin(ids)]
    sorted_df = filtered_movies.sort_values(by='movieId', key=lambda x: x.map({v: i for i, v in enumerate(ids)}))
    sorted_df["similarity score"]=sorted.values
    return sorted_df





