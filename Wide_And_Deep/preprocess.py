import pandas as pd
import os


if __name__ == '__main__':
    path = os.getcwd()
    movies = pd.read_csv(os.path.join(path, "data", 'movies.csv'))
    ratings = pd.read_csv(os.path.join(path, "data", 'ratings.csv'))

    merged_df = pd.merge(ratings, movies, on='movieId', how='inner')
    print(merged_df.columns)
    print(merged_df.head(5))
    input_file_df = merged_df[['userId', 'movieId', 'rating', 'genres']]
    input_file_df = input_file_df.rename(columns={'userId': 'userID', 'movieId': 'itemID', 'genres': 'genre'})
    input_file_df.to_csv(os.path.join(path, "processed_data", 'wide_and_deep_input_file.csv'), index=False)

    ground_truth_df = merged_df[['userId', 'movieId', 'rating', 'title', 'genres']]
    ground_truth_df.to_csv(os.path.join(path, "processed_data", 'ground_truth.csv'), index=False)

