import pandas as pd

def getUserFileFormat():
    user = pd.read_csv('ml-yahoo/ydata-ymovies-user-demographics-v1_0.txt', delimiter='\t', header=None,
                       names=['userid', 'birth_year', 'gender'])
    user = user.drop(['birth_year'], axis=1)
    user.to_csv('ml-yahoo/users.csv', index=False)

def getMovieFileFormat():
    movie = pd.read_csv('ml-yahoo/movie_db_yoda', delimiter='\t', header=None, encoding='latin1')
    # names=['movieid', 'title', 'synopsis','runtime','MPAA_rating','reasons_MPAA_rating','releasedate',
    #       'distributor','poster_URL','genres','directors','director_ids','crew_members',
    #       'crew_ids','crew_types','actors','actors_id','avg_critic_rating',
    #       'num_critic_ratings', 'no_awards_won','no_awards_nominated','list_wining_awards',
    #       'list_awards_nominated','rating_Movie_Mom','review_Movie_Mom',
    #       'list_review_summaries', 'list_anonymized_review_owners','captions',
    #       'Greg_Preview_URL','DVDreview_URL','GNPP','average_rating', 'no_ratedusers'])

    ####### genre listed as 10 index
    movie = movie.iloc[:, [0, 1, 10]]
    movie.columns = ['movieid', 'title', 'genres']
    # print(movie)
    # movie.to_csv('yahoo_movie_dataset/OLD_movies.csv', index=False)

    # movie['Genre'] = movie['genres'].astype(str).apply(lambda x: x.strip().split('|'))
    # movie['Genre'] = movie['Genre'].astype(str).apply(lambda x: x.strip().split('/'))
    # movie.loc[movie['Genre'] == 'Kids, Family', 'Genre'] = "Children's"
    # movie_genre = movie_genre.strip().split('|')

    # movie['Genre'] = movie['genres'].str.replace('Kids/Family', "Family")
    # movie['Genre'] = movie['genres'].str.replace('Musical/Performing Arts', "Musical")
    # movie['Genre'] = movie['genres'].str.replace('Crime/Gangster', "Crime")
    # movie['Genre'] = movie['Genre'].str.replace('Art/Foreign', "Western")
    # movie['Genre'] = movie['Genre'].str.replace('Science Fiction', "Sci-Fi")
    # movie['Genre'] = movie['Genre'].str.replace('/', '|')

    movie['Genre'] = movie['genres'] \
        .str.replace('Kids/Family', 'Children\'s') \
        .str.replace('Musical/Performing Arts', 'Musical') \
        .str.replace('Crime/Gangster', 'Crime') \
        .str.replace('Art/Foreign', 'Western') \
        .str.replace('Science Fiction', 'Sci-Fi') \
        .str.replace('Suspense/Horror', 'Horror') \
        .str.replace('/', '|')
    # print(movie['Genre'])
    # print(movie_genre)
    movie = movie[~movie['genres'].str.contains("~Delete|Adult Audience|Action and Adventure")]
    movie.to_csv('ml-yahoo/movies.csv', index=False)

    print(len(movie['Genre']))
    ## count the number of movies in the DataFrame where the genre is 'Special Interest'
    special_interest_count = len(movie[movie['Genre'] == 'Special Interest'])
    print("Number of movies with genre 'Special Interest':", special_interest_count)
    N_count = len(movie[movie['Genre'] == '\\N'])
    print(N_count)

def mergetraintestFile():

    train = pd.read_csv('ml-yahoo/ydata-ymovies-user-movie-ratings-train-v1_0.txt', delimiter='\t',
                        header=None, names=['userid', 'movieid', 'yahoorating', 'rating'])
#, names=['userid', 'movieid', 'yahoorating', 'rating']
    test = pd.read_csv('ml-yahoo/ydata-ymovies-user-movie-ratings-test-v1_0.txt', delimiter='\t',
                        header=None, names=['userid', 'movieid', 'yahoorating', 'rating'])

    mergeDF = pd.concat([train, test], axis =0)
    mergeDF = mergeDF.sort_values(by='userid')
    print(mergeDF)
    mergeDF.to_csv('ml-yahoo/ratings.csv', index=False)

def formatMovieID():

    # ----- there is 11915 unique movies
    # ----- only 9293 movies contains genre data (old 9249)
    movieDF_beforefilter = pd.read_csv('ml-yahoo/movies.csv')
    #movieDF = pd.read_csv('yahoo_movie_dataset/OLD_movies.csv')
    ratingDF = pd.read_csv('ml-yahoo/ratings.csv')

    movieDF = movieDF_beforefilter[~movieDF_beforefilter['Genre'].str.contains(r'\\N')]

    # Get unique movie IDs from both DataFrames
    movie_ids_movieDF = set(movieDF['movieid'])
    movie_ids_ratingDF = set(ratingDF['movieid'])

    common_movie_ids = sorted(movie_ids_movieDF & movie_ids_ratingDF)
    print('common_movie_ids len: ', len(common_movie_ids))


    # Get unique movie IDs from both DataFrames
    #all_movie_ids = sorted(set(movieDF['movieid']).union(set(ratingDF['movieid'])))
    #print('all_movie_ids total: ', len(all_movie_ids))

    # Filter rows in movieDF and ratingDF to keep only those with common movie IDs
    movieDF_filtered = movieDF.loc[movieDF['movieid'].isin(common_movie_ids)].copy()
    ratingDF_filtered = ratingDF.loc[ratingDF['movieid'].isin(common_movie_ids)].copy()

    # Create a mapping dictionary to map movie IDs to integer values
    id_mapping = {mid: idx + 1 for idx, mid in enumerate(common_movie_ids)}

    # Map movieid to mid_c in both dataframes
    movieDF_filtered['mid_c'] = movieDF_filtered['movieid'].map(id_mapping)
    ratingDF_filtered['mid_c'] = ratingDF_filtered['movieid'].map(id_mapping)

    print(len(movieDF_filtered))
    movieDF_filtered.to_csv('ml-yahoo/update_movie.csv', index=False)
    ratingDF_filtered.to_csv('ml-yahoo/update_ratings.csv', index=False)

# def formatUserID():
#
#     # --- after deleting \N genres movies, some users don't exits in rating.csv file
#     # --- so we filter again the user data and delete those users whose rating are not
#     # --- exits in rating.csv file
#     # --- initially there was 7642 users (users.csv) and now we have 7637 (update_users.csv)
#
#     userDF = pd.read_csv('ml-yahoo/users.csv')
#     ratingDF = pd.read_csv('ml-yahoo/update_ratings.csv')
#
#     # Get unique user IDs from both DataFrames
#     user_ids_userDF = set(userDF['userid'])
#     user_ids_ratingDF = set(ratingDF['userid'])
#
#     common_user_ids = sorted(user_ids_userDF & user_ids_ratingDF)
#     print('common_user_ids len: ', len(common_user_ids))
#
#     # Filter rows in movieDF and ratingDF to keep only those with common movie IDs
#     userDF_filtered = userDF.loc[userDF['userid'].isin(common_user_ids)].copy()
#     ratingDF_filtered = ratingDF.loc[ratingDF['userid'].isin(common_user_ids)].copy()
#
#     # Create a mapping dictionary to map movie IDs to integer values
#     id_mapping = {mid: idx + 1 for idx, mid in enumerate(common_user_ids)}
#
#     # Map movieid to uid_c in both dataframes
#     userDF_filtered['uid_c'] = userDF_filtered['userid'].map(id_mapping)
#     ratingDF_filtered['uid_c'] = ratingDF_filtered['userid'].map(id_mapping)
#
#     userDF_filtered.to_csv('ml-yahoo/update_users.csv', index=False)
#     ratingDF_filtered.to_csv('ml-yahoo/update_ratings.csv', index=False)

# Filter users who have rated at least 10 movies
def formatUserID():

    # Load the user and rating data
    userDF = pd.read_csv('ml-yahoo/users.csv')
    ratingDF = pd.read_csv('ml-yahoo/update_ratings(before 10 filter).csv')

    # Filter users who have rated at least 10 movies
    user_rating_counts = ratingDF['userid'].value_counts()
    users_with_min_ratings = user_rating_counts[user_rating_counts >= 20].index

    # Get unique user IDs from both DataFrames
    user_ids_userDF = set(userDF['userid'])
    user_ids_ratingDF = set(users_with_min_ratings)
    user_ids_rating = set(ratingDF['userid'])
    print(f'initial user id len: {len(user_ids_userDF)}, ids in rating: {len(user_ids_rating)}')
    print(f'after 10 filter: {len(user_ids_ratingDF)}')

    # Get the common user IDs between the filtered rating and user DataFrames
    common_user_ids = sorted(user_ids_userDF & user_ids_ratingDF)
    print('common_user_ids len: ', len(common_user_ids))

    # Filter rows in userDF and ratingDF to keep only those with common user IDs
    userDF_filtered = userDF.loc[userDF['userid'].isin(common_user_ids)].copy()
    ratingDF_filtered = ratingDF.loc[ratingDF['userid'].isin(common_user_ids)].copy()

    # Create a mapping dictionary to map user IDs to integer values
    id_mapping = {uid: idx + 1 for idx, uid in enumerate(common_user_ids)}

    # Map userid to uid_c in both dataframes
    userDF_filtered['uid_c'] = userDF_filtered['userid'].map(id_mapping)
    ratingDF_filtered['uid_c'] = ratingDF_filtered['userid'].map(id_mapping)

    # Save the filtered data to new CSV files
    userDF_filtered.to_csv('ml-yahoo/update_users.csv', index=False)
    ratingDF_filtered.to_csv('ml-yahoo/update_ratings.csv', index=False)

def formatMovieID_afterFilter():

    movieDF_beforefilter = pd.read_csv('ml-yahoo/update_movie(before 10 filter).csv')
    #movieDF = pd.read_csv('yahoo_movie_dataset/OLD_movies.csv')
    ratingDF = pd.read_csv('ml-yahoo/update_ratings.csv')

    # movieDF = movieDF_beforefilter[~movieDF_beforefilter['Genre'].str.contains(r'\\N')]

    # Get unique movie IDs from both DataFrames
    movie_ids_movieDF = set(movieDF_beforefilter['movieid'])
    movie_ids_ratingDF = set(ratingDF['movieid'])

    common_movie_ids = sorted(movie_ids_movieDF & movie_ids_ratingDF)
    print('common_movie_ids len: ', len(common_movie_ids))


    # Get unique movie IDs from both DataFrames
    #all_movie_ids = sorted(set(movieDF['movieid']).union(set(ratingDF['movieid'])))
    #print('all_movie_ids total: ', len(all_movie_ids))

    # Filter rows in movieDF and ratingDF to keep only those with common movie IDs
    movieDF_filtered = movieDF_beforefilter.loc[movieDF_beforefilter['movieid'].isin(common_movie_ids)].copy()
    ratingDF_filtered = ratingDF.loc[ratingDF['movieid'].isin(common_movie_ids)].copy()

    # Create a mapping dictionary to map movie IDs to integer values
    id_mapping = {mid: idx + 1 for idx, mid in enumerate(common_movie_ids)}

    # Map movieid to mid_c in both dataframes
    movieDF_filtered['mid_c'] = movieDF_filtered['movieid'].map(id_mapping)
    ratingDF_filtered['mid_c'] = ratingDF_filtered['movieid'].map(id_mapping)

    print(len(movieDF_filtered))
    movieDF_filtered.to_csv('ml-yahoo/update_movie.csv', index=False)
    ratingDF_filtered.to_csv('ml-yahoo/update_ratings.csv', index=False)

def mergeFiles():

    user_DF = pd.read_csv('ml-yahoo/update_users.csv')
    movie_DF = pd.read_csv('ml-yahoo/update_movie.csv')
    rating_DF = pd.read_csv('ml-yahoo/update_ratings.csv')

    mergeMovieRating = pd.merge(rating_DF[['uid_c', 'mid_c', 'rating']],
                                movie_DF[['mid_c', 'Genre']], on='mid_c')
    #print(mergeMovieRating)

    mergeAll = pd.merge(mergeMovieRating, user_DF[['uid_c','gender']], on='uid_c')


    #mergeUser = user_DF[user_DF['userid'].isin(mergeAll['userid'])]

    print(len(mergeAll))
    mergeAll.to_csv('ml-yahoo/yahoo_mergerating.csv', index=False)
    #mergeUser.to_csv('yahoo_movie_dataset/update_users.csv', index=False)

def ShrinkMovieFile():
    movies_df = pd.read_csv('ml-yahoo/update_movie.csv')
    print(movies_df)
    movies_df_subset = movies_df[['mid_c', 'title', 'Genre']]
    print(movies_df_subset)
    # movies_df_subset.rename(columns={'mid_c': 'movieId', 'Genre': 'genres'}, inplace=True)
    movies_df_subset.to_csv('ml-yahoo/shrink_movies.dat', index=False, header=None)
    print(movies_df_subset)

# ------------- function call -------------------
#mergetraintestFile()

# --- this getMovieFileFormat() function shrink the movie file and generate new file with required column
# --- and the newly created file's genre is modified by erasing uneven genres
#getMovieFileFormat()

# formatMovieID()
#formatUserID()
#formatMovieID_afterFilter()
# mergeFiles() #207307 # after 20 filter: 142836
ShrinkMovieFile()
