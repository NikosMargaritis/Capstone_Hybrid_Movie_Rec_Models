import tensorflow as tf
from recommenders.utils import tf_utils, gpu_utils, plot
import sys
import pandas as pd
import os
import data_preprocess as dp

if __name__ == '__main__':
    print(f"System version: {sys.version}")
    print(f"Tensorflow version: {tf.__version__}")
    print(f"GPUs: {gpu_utils.get_gpu_info()[0]}")

    path = os.getcwd()
    movies = pd.read_csv(os.path.join(path, "data", 'movies.csv'))
    ratings = pd.read_csv(os.path.join(path, "data", 'ratings.csv'))

    movies_one_hot = dp.create_one_hot_encodings(movies)
    print(movies_one_hot.head(5))

    merged_df = pd.merge(ratings, movies_one_hot, on='movieId', how='inner')
    print(merged_df.columns)
    print(merged_df.head(5))
    merged_df = merged_df[['userId', 'movieId', 'rating', 'features']]
    merged_df.to_csv("merged.csv")