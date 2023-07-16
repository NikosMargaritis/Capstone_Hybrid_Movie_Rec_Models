import pandas as pd
import os


if __name__ == '__main__':
    path = os.getcwd()
    movies = pd.read_csv(os.path.join(path, "data", 'movies.csv'))
    predictions_df = pd.read_csv(os.path.join(path, "processed_data", 'wide_and_deep_predictions.csv'))
    predictions_df = predictions_df.rename(columns={'itemID': 'movieId'})

    merged_df = pd.merge(predictions_df, movies, on='movieId', how='inner')
    print(merged_df.columns)
    print(merged_df.head(5))
    output_file_df = merged_df[['userID', 'movieId', 'scaled_predictions', 'title', 'genres']]
    output_file_df = output_file_df.rename(columns={'userID': 'userId'})
    output_file_df.to_csv(os.path.join(path, "processed_data", 'wide_and_deep_processed_predictions.csv'), index=False)

