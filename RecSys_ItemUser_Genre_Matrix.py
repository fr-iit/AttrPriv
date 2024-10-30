import pandas as pd
import RecSys_DataLoader as DL
import numpy as np

'''
Notes: 
1. Load the movie information file (adjust the delimiter according to your .dat file format)
2. Remove the header from the movie files
3. List of all possible genres (make sure this list matches your data)
'''

genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def Item_Genre_Matrix(data):

    if data == '1m':
        path = 'ml-1m/'
        user_item_matrix = DL.load_user_item_matrix_1m()
        movies_df = pd.read_csv('ml-1m/movies.dat', delimiter=',', engine='python', header=None, names=['movieId', 'title', 'genres'])

    elif data == '100k':
        path = 'ml-100k/'
        user_item_matrix = DL.load_user_item_matrix_100k()
        movies_df = pd.read_csv('ml-100k/movies.dat', delimiter=':', engine='python', header=None, names=['movieId', 'title', 'genres'])

    elif data == 'yahoo':
        path = 'ml-yahoo/'
        user_item_matrix = DL.load_user_item_matrix_yahoo()
        movies_df = pd.read_csv('ml-yahoo/shrink_movies.dat', delimiter=',', engine='python', header=None, names=['movieId', 'title', 'genres'])


    itemgen_matrix = pd.DataFrame(0, index=movies_df['movieId'], columns=genres)

    for index, row in movies_df.iterrows():
        if row['genres'] is not None and row['genres'] != '':
            item_gen = row['genres'].split('|')
        else:
            continue  # Skip rows with no genres

        for gen in item_gen:
            if gen in genres:
                itemgen_matrix.at[row['movieId'], gen] = 1

    # Save the item-genre matrix to a .dat file using '::' as the delimiter (you can change the delimiter as needed)

    itemgen_matrix.to_csv(path+'Item_Genre_Matrix.dat', sep=',', header=True, index=True)

    return itemgen_matrix, path, user_item_matrix


def User_Genre_Preference(item_genre_matrix, path, rating_matrix):
    # Load data
    # if data == '1m':
    #     # path = 'ml-1m/'
    #     user_item_matrix = DL.load_user_item_matrix_1m()
    # elif data == '100k':
    #     # path = 'ml-100k/'
    #     user_item_matrix = DL.load_user_item_matrix_100k()
    # elif data == 'yahoo':
    #     # path = ''
    #     user_item_matrix = DL.load_user_item_matrix_yahoo()

    # Assuming this function loads a user-item rating matrix
    user_item_matrix = rating_matrix
    num_users = user_item_matrix.shape[0]

    # Create dictionaries to store preferences and counts
    user_genre_pref_dict = {}
    user_genre_count_dict = {}

    for user_id in range(num_users):
        # if user_id ==5:
        #     break
        user_genre_pref_dict[user_id] = {genre: 0.0 for genre in genres}
        user_genre_count_dict[user_id] = {genre: 0 for genre in genres}

        user_rating = user_item_matrix[user_id, :]
        rated_item_indices = np.where(user_rating > 0)[0]

        for genre in genres:
            numerator = 0.0
            denominator_genk = 0
            denominator_total_gen = 0

            for item_index in rated_item_indices:
                rating = user_rating[item_index]
                item_id = item_index  # Assuming item_index corresponds to the movie ID

                if item_id in item_genre_matrix.index:
                    # Calculate total genres for the movie
                    total_gen = item_genre_matrix.loc[item_id].sum()  # Count total genres the movie belongs to

                    # Check if the movie belongs to the current genre
                    if item_genre_matrix.at[item_id, genre] == 1:
                        numerator += (rating/total_gen) #rating#
                        user_genre_count_dict[user_id][genre] += 1  # Update the count for the genre

                    # denominator_total_gen += total_gen
                    denominator_genk += item_genre_matrix.at[item_id, genre]

                    print(f"User {user_id}, item: {item_id}, Genre {genre}: Numerator = {numerator}, Denominator_genk = {denominator_genk}")

            # Update the preference matrix only if the denominator is valid
            if denominator_genk > 0:
                # user_genre_pref_dict[user_id][genre] = numerator / (denominator_total_gen * denominator_genk)
                user_genre_pref_dict[user_id][genre] = round(numerator / denominator_genk)#numerator / denominator_genk
                print(f"pref. mat: {user_genre_pref_dict[user_id][genre]}")
            else:
                user_genre_pref_dict[user_id][genre] = 0  # If no valid ratings, keep it as zero

    # Convert dictionaries to DataFrames
    user_genre_matrix = pd.DataFrame.from_dict(user_genre_pref_dict, orient='index')
    # user_genre_count = pd.DataFrame.from_dict(user_genre_count_dict, orient='index')
    print(user_genre_matrix)

    # Save the matrices
    user_genre_matrix.to_csv(path+'user_genre_matrix_round.dat', sep=',', header=True, index=True)
    # user_genre_count.to_csv('ml-1m/user_genre_count.dat', sep=',', header=True, index=True)

    return user_genre_matrix, 0#user_genre_count


# Assuming you have loaded or created the item_genre_matrix
data = '100k'
matrix, path, rating_matrix = Item_Genre_Matrix(data)
user_genre_matrix, user_genre_count = User_Genre_Preference(matrix, path, rating_matrix)