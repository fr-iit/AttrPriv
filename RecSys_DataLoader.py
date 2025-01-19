import numpy as np
import pandas as pd
import csv

# dataset: yahoo
def load_user_item_matrix_yahoo():

    movies = set()  
    users = set()  # Using set to automatically deduplicate
    ratings = []

    with open('ml-yahoo/yahoo_mergerating.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header

        for row in reader:
            userid, movieid, Rating, Genres, gender = row

            user_id = int(userid)
            movie_id = int(movieid)
            rating = float(Rating)

            movies.add(movie_id)
            users.add(user_id)
            ratings.append((user_id, movie_id, rating))

    # Map unique user and movie IDs to their indices
    num_unique_users = len(users)
    num_unique_movies = len(movies)

    print(f'Number of unique users: {num_unique_users}, Number of unique movies: {num_unique_movies}')

    # Create a user-item matrix with dimensions (num_users, num_movies)
    df = np.zeros(shape=(num_unique_users, num_unique_movies))

    # Fill the matrix with ratings
    for user_id, movie_id, rating in ratings:
        df[user_id - 1, movie_id - 1] = rating  # Subtracting 1 to align with 0-indexing

    # Calculate density
    count_non_zero = np.count_nonzero(df)
    density = (count_non_zero / df.size) * 100

    print(f'Yahoo Data Density: {density:.2f}%')

    return df

def load_gender_vector_yahoo():
    gender = []
    m_count = 0
    f_count = 0
    with open('ml-yahoo/update_users.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            userid,gender_val, uid = row
            if gender_val == 'm':
                gender.append(0)
                m_count +=1
            else:
                gender.append(1)
                f_count +=1
    print('male count: ', m_count, " & female count: ", f_count)

    return np.asarray(gender)

# for 20 rating filter: max_user=2837, max_item=8584,
def load_user_item_matrix_yahoo_masked(max_user=2837, max_item=8584, file_index=-1):

    files = [
        # #----- BlueMe: for 20 rating filter
        # "ml-yahoo/BlurMe/Threshold20Rating_blurMe_Yahoo_obfuscated_0.05_greedy_avg.dat" 
        
        #-----PerBlur ---- no removal ---- avg_greedy
        # "ml-yahoo/PerBlur/PBNoRemoval_Top100Indicative_avg_greedy_0.05.dat"  
        
        # -----PerBlur with removal ----
        # "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.05.dat" 
        
        # ----- SBlur: no removal: indicative top100
        # 'ml-yahoo/SBlur/SBlur_avg_greedy_0.05_2.dat'
        
        # ----- SBlur with removal: indicative top100
        # 'ml-yahoo/SBlur/SBlur_Removal_avg_greedy_0.05_2.dat'  
    ]

    df = np.zeros(shape=(max_user, max_item))
    for file in files:
        print(f"Using file: {file} & index: {files[file_index]}")

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    # Calculate density
    count_non_zero = np.count_nonzero(df)
    density = (count_non_zero / df.size) * 100

    print(f'Yahoo Obfuscated Data Density: {density:.2f}%')

    return df

def gender_user_dictionary_yahoo():
    gender_dict = {}
    with open("ml-yahoo/update_users.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            uid, gender, userid = row
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    #print(gender_dict)
    return gender_dict

def load_user_item_matrix_yahoo_Impute(max_user=2837, max_item=8584):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-yahoo/Dist/combine_knn_imputed_user_item_matrix_30_top100.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df
# --- end yahoo

# dataset: ml100k
def load_user_item_matrix_100k(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("ml-100k/u.data", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_100k_Impute(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))

    with open("ml-100k/Dist/combine_knn_imputed_user_item_matrix_30_top100.dat", 'r') as f:  
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")  
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df

def load_user_item_matrix_100k_masked(max_user=943, max_item=1682, file_index=-1):

    df = np.zeros(shape=(max_user, max_item))
    masked_files = [
        # Add path to your file obfuscated by BlurMe, we start from #0

        #blurme: test with LR lm & lf items (top)  + non common items -> initial AUC: 0.79
        # "ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.05_greedy_avg.dat" 
        
        #PerBlur: no removal
        # "ml-100k/PerBlur/PerBlur_ML100k_avg_greedy_0.05_2.dat" 
        
        # PerBlur: with removal
        # "ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100_avg_greedy_0.05_2_strategic.dat"  

        'ml-100k/SBlur/SBlur_Removal_pred_greedy_0.05.dat'
    ]

    print(masked_files[file_index])
    with open(masked_files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df

def load_gender_vector_100k(max_user=943):

    gender_vec = []
    m = 0
    ff = 0
    with open("ml-100k/u.user", 'r') as f:
        for line in f.readlines():
            userid, age, gender, occupation, zipcode = line.split("|")
            if gender == "M":
                m += 1
                gender_vec.append(0)
            else:
                ff += 1
                gender_vec.append(1)
    print(f'male: {m}, female: {ff}')
    return np.asarray(gender_vec)

def gender_user_dictionary_100k():
    gender_dict = {}
    with open("ml-100k/u.user", 'r') as f:
        for line in f.readlines():
            userid, age, gender, occupation, zipcode = line.split("|")
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    return gender_dict

# --- end ml100k
# dataset: ml1m
def load_user_item_matrix_1m(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_1m_Impute(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))

    with open("ml-1m/Dist/combine_knn_imputed_user_item_matrix_30.dat", 'r') as f:
    # with open("ml-1m/Dist/knn_imputed_user_item_matrix_30.dat", 'r') as f:
    # with open("ml-1m/TrainingSet_users_KNN_fancy_imputation_ml-1m_k_30.dat", 'r') as f: # --- this is used in perblur
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_1m_masked(max_user=6040, max_item=3952, file_index=-1):

    files = [
        
        #----- BlueMe
        # "ml-1m/BlurMe/TrainingSet_blurMe_ML1M_obfuscated_0.05_greedy_avg_top-1.dat"
        
        #-----PerBlur ---- no removal ---- avg_greedy
        # "ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2.dat" 
        
        # -----PerBlur ---- no removal ---- pred_greedy
        # "ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_pred_greedy_0.05_2.dat" 
        
        # ----- SBlur with removal: indicative top100
        # "ml-1m/SBlur/SBlur_withRemoval_pred_greedy_0.05_2.dat"         
    ]

    df = np.zeros(shape=(max_user, max_item))
    for file in files:
        print(f"Using file: {file} & index: {files[file_index]}")

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_gender_vector_1m(max_user=6040):

    gender_vec = []
    m = 0
    ff = 0
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:max_user]:
            user_id, gender, age, occ, postcode = line.split("::")
            if gender == "M":
                m +=1
                gender_vec.append(0)
            else:
                ff +=1
                gender_vec.append(1)
    print(f'ml-1m; male: {m}, female: {ff}')
    return np.asarray(gender_vec)

def gender_user_dictionary_1m():
    gender_dict = {}
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            userid, gender, age, occupation, zipcode = line.split("::")
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    return gender_dict
# --- end ml1m

def DensityCount(data = 'yahoo'):

    if data == '1m':
        X = load_user_item_matrix_1m()
    elif data == '100k':
        X = load_user_item_matrix_100k()
    elif data == 'yahoo':
        X = load_user_item_matrix_yahoo()
        print(f'user: {X.shape[0]}, items: {X.shape[1]}')

    total_entries = X.shape[0] * X.shape[1]
    no_of_ratings = np.count_nonzero(X)
    density = (no_of_ratings/total_entries) * 100

    print(f"data: {data} , density: {density}")


# load_gender_vector_100k()
# load_gender_vector_1m()
# DensityCount()
