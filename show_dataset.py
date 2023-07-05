import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.sparse import csr_matrix
#from sklearn.neighbors import NearestNeighbors
#from sklearn.model_selection import train_test_split

DATASET_LINK = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

overall_stats = pd.read_csv('ml-100k/u.info', header=None)
print("Details of users, items and ratings involved in the loaded movielens dataset: ", list(overall_stats[0]))

## same item id is same as movie id, item id column is renamed as movie id
column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
print(dataset.head())

d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')

items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
items_dataset.to_csv("items_nick.csv")