import pandas as pd
import numpy as np

def create_one_hot_encodings(movies_df):
    movies_df = movies_df.dropna(subset=['genres'])
    genres = movies_df['genres'].values.tolist()
    all_genres = []
    for item in genres:
        lista = item.split('|')
        for item2 in lista:
            if item2 not in all_genres:
                all_genres.append(item2)
    print(len(all_genres))
    all_genres = sorted(all_genres)
    print(all_genres)

    one_hot_encodings_column = []
    for genre in genres:
        print(genre)
        temp = [0 for _ in range(len(all_genres))]
        lista = genre.split('|')
        for item in lista:
            pos = all_genres.index(item)
            temp[pos] = 1
        print(temp)
        one_hot_encodings_column.append(temp)
    movies_df['features'] = one_hot_encodings_column
    # movies_df.to_csv("new_movies.csv", index=False)
    return movies_df
