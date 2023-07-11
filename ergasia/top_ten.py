import pandas as pd
import numpy as np

if __name__ == '__main__':
    movies_df = pd.read_csv("data/movies.csv")
    predictions_df = pd.read_csv("preds_nick.csv")
    users = predictions_df['userID'].drop_duplicates().values.tolist()
    print(sorted(users))

    print(predictions_df.columns)

    yo = predictions_df.loc[predictions_df['userID'] == 49]
    yo = yo.sort_values(by=['prediction'], ascending=False)
    print(yo.head(10))

    Row_list = []

    # Iterate over each row

    for index, rows in yo.iterrows():
        # Create list for the current row
        my_list = [rows.userID, rows.itemID, rows.prediction]

        # append the list to the final list
        Row_list.append(my_list)
        if len(Row_list) == 10:
            break
    print(["UserID", "MovieID", "Prediction", "Title", "Genres"])
    for item in Row_list:
        movie = movies_df.loc[movies_df['movieId'] == item[1]]
        item.extend(movie.title.values.tolist())
        item.extend(movie.genres.values.tolist())
        print(item)
