import numpy as np
import pandas as pd
import csv

# dataset: yahoo
def load_user_item_matrix_yahoo():

    movies = set()  # Using set to automatically deduplicate
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
# Number of unique users: 7434, Number of unique movies: 9244
# Yahoo Data Density: 0.30%
# ok = load_user_item_matrix_yahoo()

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

# for actual data: max_user=7434, max_item=9244,
# for 20 rating filter: max_user=2837, max_item=8584,
def load_user_item_matrix_yahoo_masked(max_user=2837, max_item=8584, file_index=-1):

    files = [
            # DD : Data Density
            #----- BlueMe
            #"ml-yahoo/BlurMe/TrainingSet_blurMe_Yahoo_obfuscated_0.01_greedy_avg.dat", #AUC: 0.74 ± 0.01; DD: 0.30%
            #"ml-yahoo/BlurMe/TrainingSet_blurMe_Yahoo_obfuscated_0.05_greedy_avg.dat" #AUC: 0.71 ± 0.02; DD: 0.31%

            # for 20 rating filter
            #"ml-yahoo/BlurMe/Threshold20Rating_blurMe_Yahoo_obfuscated_0.05_greedy_avg.dat" #AUC: 0.69 ± 0.05; DD: 0.62%
            #"ml-yahoo/BlurMe/Threshold20Rating_blurMe_Yahoo_obfuscated_0.01_greedy_avg.dat"  # AUC: 0.77 ± 0.03; DD: 0.60%
            #"ml-yahoo/BlurMe/Threshold20Rating_blurMe_Yahoo_obfuscated_0.1_greedy_avg.dat" # AUC: 0.64 ± 0.06; DD: 0.65%

            #-----PerBlur ---- no removal ---- avg_greedy
            #"ml-yahoo/PerBlur/PBNoRemoval_Top100Indicative_avg_greedy_0.01.dat" #AUC: 0.77 ± 0.02
            #"ml-yahoo/PerBlur/PBNoRemoval_Top100Indicative_avg_greedy_0.02.dat"  # AUC: 0.76 ± 0.02
            #"ml-yahoo/PerBlur/PBNoRemoval_Top100Indicative_avg_greedy_0.05.dat"  # AUC: 0.72 ± 0.02
            #"ml-yahoo/PerBlur/PBNoRemoval_Top100Indicative_avg_greedy_0.1.dat"  # AUC: 0.72 ± 0.02
            # "ml-yahoo/PerBlur/PBNoRemoval_Top40Indicative_avg_greedy_0.05.dat"

            # -----PerBlur ---- no removal ---- pred_greedy

            # -----PerBlur with removal ----
            # "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.05.dat" # AUC: 0.62 ± 0.03
            # "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.01.dat" # AUC: 0.69 ± 0.03
            # "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.02.dat" # AUC: 0.68 ± 0.04
            # "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.1.dat" # AUC: 0.59 ± 0.06
        "ml-yahoo/PerBlur/PB_Top100Indicative_avg_greedy_0.15.dat"  # AUC: 0.59 ± 0.06

        # ---- perblur: top50:

            # ----- SBlur: no removal: indicative top100

            # ----- SBlur with removal: indicative top100

    ]
    #id_index, _ = load_movie_id_index_dict()
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

    with open("ml-1m/Dist/combine_knn_imputed_user_item_matrix_30.dat", 'r') as f:  # All_allUsers_KNN_fancy_imputation_100k_k_30
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")  # All_Opposite_Gender_KNN_fancy_imputation_1m_k_30VF || All_allUsers_KNN_fancy_imputation_1m_k_30
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df

def load_user_item_matrix_100k_masked(max_user=943, max_item=1682, file_index=-1):

    df = np.zeros(shape=(max_user, max_item))
    masked_files = [
        # Add path to your file obfuscated by BlurMe, we start from #0

        #blurme: test with LR lm & lf items (top)  + non common items -> initial AUC: 0.79
        #"ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.05_greedy_avg.dat" #-> .58
        #"ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.01_greedy_avg.dat" # .73
        #"ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.02_greedy_avg.dat" # .70
        #"ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.1_greedy_avg.dat" # .43
        #"ml-100k/blurMe/TrainingSet_blurMe_ML100k_obfuscated_0.07_greedy_avg.dat" # .52

        #PerBlur: no removal
        #"ml-100k/PerBlur/PerBlur_ML100k_avg_greedy_0.1_2.dat"  # AUC: 0.55 +-.06
        #"ml-100k/PerBlur/PerBlur_ML100k_avg_greedy_0.05_2.dat" #AUC: 0.69 +-.04
        #"ml-100k/PerBlur/PerBlur_ML100k_avg_greedy_0.02_2.dat"  # AUC: 0.76 +-.04
        #"ml-100k/PerBlur/PerBlur_ML100k_avg_greedy_0.01_2.dat"  # AUC: 0.77 +-.04

        # PerBlur: with removal
        #"ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100IndicativeItems_avg_greedy_0.1_2_strategic.dat"  # AUC: 0.55 +-.06
        #"ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100IndicativeItems_avg_greedy_0.01_2_strategic.dat" #AUC: 0.68 +-.04
        #"ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100IndicativeItems_avg_greedy_0.02_2_strategic.dat"  # AUC: 0.63 +-.04
        #"ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100_avg_greedy_0.05_2_strategic.dat"  # AUC: 0.47 +-.04

        # ---- perblur : ml100k: no removal ----- according to new lm lf
        #"ml-100k/PerBlur/OwnTest/PerBlur_New_avg_greedy_0.05_2.dat" #AUC: 0.58 +-.05
        #"ml-100k/PerBlur/OwnTest/PerBlur_New_pred_greedy_0.05_2.dat"
        #"ml-100k/PerBlur/OwnTest/PerBlur_New_pred_greedy_0.1_2.dat"
        #"ml-100k/PerBlur/OwnTest/PerBlur_withRemove_pred_greedy_0.05_2.dat"
        #"ml-100k/PerBlur/thresh20_PerBlur_ML100k_obfuscated_Top100_avg_greedy_0.05_2_strategic.dat"

        # ---- sblur :ml100k: only item addition
        # "ml-100k/PerBlur/OwnTest/SBlur_pred_greedy_0.05_2.dat" #AUC: 0.60 ± 0.05
        # "ml-100k/PerBlur/OwnTest/SBlur_pred_greedy_0.02_2.dat" #AUC: 0.69 ± 0.05
        #"ml-100k/PerBlur/OwnTest/SBlur_pred_greedy_0.01_2.dat" #AUC: 0.71 ± 0.04
        #"ml-100k/PerBlur/OwnTest/SBlur_pred_greedy_0.1_2.dat" #AUC: 0.46 ± 0.06

        # ---- sblur :ml100k: with removal
        #"ml-100k/PerBlur/OwnTest/SBlur_withRemove_pred_greedy_0.05_2.dat" #AUC: 0.43 ± 0.05
        #"ml-100k/PerBlur/OwnTest/SBlur_withRemove_pred_greedy_0.02_2.dat" 0.61 ± 0.05
        #"ml-100k/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.01_2.dat" #0.66 ± 0.04
        "ml-100k/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.1_2.dat"



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
    # with open("data/ml-100k/userObs.csv", 'r') as f:
    with open("ml-100k/u.user", 'r') as f:
        # for line in f.readlines()[1:]:
        #     if len(line) < 2:
        #         continue
        #     else:
        #         userid, age, gender, occupation, zipcode = line.split(", ")
        #         if gender == "M":
        #             gender_vec.append(0)
        #         else:
        #             gender_vec.append(1)

        for line in f.readlines():
            userid, age, gender, occupation, zipcode = line.split("|")
            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)
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
    with open("ml-1m/Dist/knn_imputed_user_item_matrix_30.dat", 'r') as f:
    #with open("ml-1m/TrainingSet_users_KNN_fancy_imputation_ml-1m_k_30.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_1m_masked(max_user=6040, max_item=3952, file_index=-1):

    files = [

             # BlurMe and BlurMore Final
            #"ml-1m/PerBlur/TrainingSet_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2_random.dat",
            #"ml-1m/Test/TrainingSet_thresh20_PerBlur_ML1M_obfuscated_Top50IndicativeItems_pred_greedy_0.02_2_strategic.dat",
            #"ml-1m/Test/TrainingSet_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.02_2_random.dat",

            #----- BlueMe
            #"ml-1m/BlurMe/TrainingSet_blurMe_ML1M_obfuscated_0.05_greedy_avg_top-1.dat", #AUC: .48 correct AUC found as per PerBlur
            #"ml-1m/BlurMe/TrainingSet_blurMe_ML1M_obfuscated_0.01_greedy_avg_top-1.dat" #AUC: .76
            #"ml-1m/BlurMe/TrainingSet_blurMe_ML1M_obfuscated_0.02_greedy_avg_top-1.dat" #AUC: .70
            #"ml-1m/BlurMe/TrainingSet_blurMe_ML1M_obfuscated_0.1_greedy_avg_top-1.dat" #AUC: .22
            #"ml-1m/BlurMe/noPerBlurTest_TrainingSet_blurMe_ML1M_obfuscated_0.1_random_avg_top-1.dat" #AUC: .81

            #-----PerBlur ---- no removal ---- avg_greedy
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2.dat" #AUC: .64
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.01_2.dat" #AUC: .81
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.02_2.dat" #AUC: .78
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.1_2.dat" #AUC: .39

            # -----PerBlur ---- no removal ---- pred_greedy
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_pred_greedy_0.01_2.dat" #AUC: .81
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_pred_greedy_0.02_2.dat" #AUC: .78
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_pred_greedy_0.05_2.dat" #AUC: .63
            #"ml-1m/PerBlur/Top-100-NoRemoval_All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_pred_greedy_0.1_2.dat"

            # -----PerBlur with removal ----
            #"ml-1m/PerBlur/Removal/thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.01_2_strategic.dat" #AUC: .66 +-2
            #"ml-1m/PerBlur/Removal/thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.02_2_strategic.dat" #AUC: .60 +-3
            #"ml-1m/PerBlur/Removal/thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2_strategic.dat" #AUC: .36 +-3
            #"ml-1m/PerBlur/Removal/thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.1_2_strategic.dat" # AUC: .14 +-2

            # ---- perblur: top50:
            #"ml-1m/PerBlur/Top50/PerBlur_ML1M_avg_greedy_0.05_2.dat"
            #"ml-1m/PerBlur/Top50/thresh20_PerBlur_ML1M_obfuscated_Top50IndicativeItems_avg_greedy_0.05_2_strategic.dat"

            # -----PerBlur ---- own test
            #"ml-1m/PerBlur/OwnTest/thresh20_PerBlur_ML1M_Top100IndicativeItems_avg_greedy_0.05_2.dat" #auc: .64
            #"ml-1m/PerBlur/OwnTest/thresh20_PerBlur_ML1M_Top100IndicativeItems_avg_greedy_0.02_2.dat" #auc: .78
            #"ml-1m/PerBlur/OwnTest/thresh20_PerBlur_ML1M_Top100IndicativeItems_pred_greedy_0.05_2.dat" #auc: .65
            #"ml-1m/PerBlur/OwnTest/PerBlur_ML1M_Top100IndicativeItems_avg_greedy_0.05_2.dat" #auc:.93 need invesrigation
            #"ml-1m/PerBlur/OwnTest/PerBlur_ML1M_avg_greedy_0.05_2.dat"
            #"ml-1m/PerBlur/OwnTest/PerBlur_New_avg_greedy_0.05_2.dat"

            # ----- SBlur: no removal: indicative top100
            #"ml-1m/PerBlur/OwnTest/SBlur_pred_greedy_0.05_2.dat" # AUC: 0.52 ± 0.04
            #"ml-1m/PerBlur/OwnTest/SBlur_pred_greedy_0.01_2.dat" #AUC: 0.74 ± 0.03
            #"ml-1m/PerBlur/OwnTest/SBlur_pred_greedy_0.02_2.dat" #AUC: 0.70 ± 0.03
            #"ml-1m/PerBlur/OwnTest/SBlur_pred_greedy_0.1_2.dat" #AUC: 0.29 ± 0.07

            # ----- SBlur with removal: indicative top100
            #"ml-1m/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.05_2.dat" #AUC: 0.23 ± 0.10
            #"ml-1m/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.02_2.dat"  # AUC: 0.55 ± 0.07
            #"ml-1m/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.01_2.dat" #AUC: 0.65 ± 0.05
            #"ml-1m/PerBlur/OwnTest/SBlur_withRemoval_pred_greedy_0.1_2.dat" #AUC: 0.05 ± 0.04


        #"ml-1m/PerBlur/TrainingSet_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2_random.dat" #AUC: .60

    ]
    #id_index, _ = load_movie_id_index_dict()
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
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:max_user]:
            user_id, gender, age, occ, postcode = line.split("::")
            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

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

#g = gender_user_dictionary_yahoo()

def DensityCount(data = '100k'):

    if data == '1m':
        X = load_user_item_matrix_1m()
    elif data == '100k':
        X = load_user_item_matrix_100k()
        X_obs = load_user_item_matrix_100k_masked()

    total_entries = X.shape[0] * X.shape[1]
    no_of_ratings = np.count_nonzero(X)
    density = (no_of_ratings/total_entries) * 100

    obs_total_entries = X_obs.shape[0] * X_obs.shape[1]
    obs_ratings_no = np.count_nonzero(X_obs)
    obs_density = (obs_ratings_no/obs_total_entries) * 100

    print(f"data: {data} , density: {density}, Obs_density: {obs_density}")
