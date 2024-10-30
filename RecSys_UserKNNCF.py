import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
import MovieLensData as MD
import RecSys_Evaluation as RSE
import pandas as pd


def split_train_test(R, test_size=0.2):

    num_users, num_items = R.shape
    train_data = []
    test_data = []

    for user in range(num_users):
        rated_items = np.nonzero(R[user, :].toarray())[1]  # Extract nonzero items
        if len(rated_items) > 9:
            train_items, test_items = train_test_split(rated_items, test_size=test_size, random_state=42)

            train_row = np.zeros(num_items)
            test_row = np.zeros(num_items)

            train_row[train_items] = R[user, train_items].toarray().flatten()
            test_row[test_items] = R[user, test_items].toarray().flatten()

            train_data.append(train_row)
            test_data.append(test_row)
        else:
            continue

    train_matrix = np.array(train_data)
    test_matrix = np.array(test_data)

    return csr_matrix(train_matrix), csr_matrix(test_matrix)

def split_testmatrix_by_gender(R_test, user_gender_dict):
    male_indices = []
    female_indices = []
    R_test_male = []
    R_test_female = []

    # Split users by gender
    for user_index, gender in user_gender_dict.items():
        if gender == "M":
            male_indices.append(user_index)
            R_test_male.append(R_test[user_index, :].toarray())
        elif gender == "F":
            female_indices.append(user_index)
            R_test_female.append(R_test[user_index, :].toarray())

    # Convert to numpy arrays
    R_test_male = csr_matrix(np.vstack(R_test_male))  # Convert to sparse matrix
    R_test_female = csr_matrix(np.vstack(R_test_female))  # Convert to sparse matrix

    return np.array(male_indices), np.array(female_indices), R_test_male, R_test_female

def compute_user_similarity(R_train):
    # Convert sparse matrix to dense
    # R_train = R_train.toarray()  # or use .todense()

    # Step 1: Compute the Euclidean distance matrix
    # pdist computes the pairwise distance, squareform converts it into a matrix

    # distance_matrix = squareform(pdist(R_train, metric='euclidean'))
    # similarity_matrix = 1 / (1 + distance_matrix)

    return cosine_similarity(R_train) #similarity_matrix #

# Function to combine rating similarity and genre similarity
def compute_combine_sim(rating_sim, genre_sim, weight=0.5):
    # Combine the similarities using weighted average
    combined_sim = (weight * rating_sim) + (1- weight) * genre_sim
    return combined_sim


def predict_userknn_rating(user_index, item_index, R_train, similarity_matrix, neighbor_count, threshold=0.15):
    # Get similarity scores for the user
    user_similarity = similarity_matrix[user_index]

    # Get the indices of the users who have rated the item
    rated_by_neighbors = R_train[:, item_index].nonzero()[0]  # Users who have rated this item
    neighbor_similarities = user_similarity[rated_by_neighbors]

    # Apply similarity threshold
    valid_neighbors = rated_by_neighbors[neighbor_similarities >= threshold]
    valid_similarities = neighbor_similarities[neighbor_similarities >= threshold]

    # If no valid neighbors pass the threshold, return user's average rating
    if len(valid_neighbors) == 0:
        return np.mean(R_train[user_index].toarray())  # Default to user's average rating

    # Sort neighbors by similarity and select top-k valid neighbors
    top_k_neighbors = np.argsort(-valid_similarities)[:neighbor_count]  # Get top-k indices of valid neighbors

    # Predict rating as weighted average of top-k neighbor ratings
    numerator = 0
    denominator = 0
    for idx in top_k_neighbors:
        neighbor_index = valid_neighbors[idx]
        neighbor_rating = R_train[neighbor_index, item_index]

        if neighbor_rating > 0:
            similarity = valid_similarities[idx]
            numerator += similarity * neighbor_rating
            denominator += abs(similarity)

    if denominator > 0:
        return numerator / denominator
    else:
        return np.mean(R_train[user_index].toarray())  # Default to user's average rating if no valid neighbors


def evaluate_userknn(R_train, R_test, similarity_matrix, k=10):

    num_users, num_items = R_test.shape
    hit_count = 0
    total_count = 0
    candidate_item = int(num_items * 0.25)
    print(f'candidate_item: {candidate_item}')

    all_true_ratings = []
    all_predicted_ratings = []
    all_ndcg_scores = []

    for user in range(num_users):
        rated_items = R_test[user, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        for item in rated_items:
            # Predict rating for each test user-item pair
            predicted_rating = predict_userknn_rating(user, item, R_train, similarity_matrix, neighbor_count=30)

            # Store true and predicted ratings for MAE calculation
            true_rating = R_test[user, item]
            all_true_ratings.append(true_rating)
            all_predicted_ratings.append(predicted_rating)

            # Check if this prediction falls in the top-k items for this user (for Hit Rate)
            non_rated_items = np.setdiff1d(np.arange(num_items), R_train[user, :].nonzero()[1])
            random_samples = np.random.choice(non_rated_items, size=(candidate_item - 1), replace=False)
            candidates = np.concatenate(([item], random_samples))
            scores = [predict_userknn_rating(user, candidate, R_train, similarity_matrix, neighbor_count=30) for candidate in candidates]

            # Collect true ratings and predicted scores for nDCG
            candidate_true_ratings = [R_test[user, candidate] for candidate in candidates]
            ndcg_score = calculate_ndcg(candidate_true_ratings, scores, k=k)
            all_ndcg_scores.append(ndcg_score)

            top_k_items = np.argsort(scores)[-k:]  # Get top-k items

            if 0 in top_k_items:  # If true item is in top-k, increment hit count
                hit_count += 1
            total_count += 1
            print(f'user: {user}, hit count: {hit_count}, total_count: {total_count}')
    hit_rate_at_k = hit_count / total_count if total_count > 0 else 0

    return hit_rate_at_k, all_true_ratings, all_predicted_ratings, all_ndcg_scores

def calculate_mae(true_ratings, predicted_ratings):
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    mae = np.mean(np.abs(true_ratings - predicted_ratings))
    return mae

def calculate_ndcg(true_ratings, predicted_scores, k=10):
    # Sort the predicted scores and corresponding true ratings in descending order of predicted scores
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_true_ratings = np.take(true_ratings, sorted_indices[:k])

    # Calculate DCG
    dcg = 0.0
    for i in range(len(sorted_true_ratings)):
        dcg += (2 ** sorted_true_ratings[i] - 1) / np.log2(i + 2)  # log2(i+2) to avoid division by 0

    # Calculate IDCG (Ideal DCG) based on the sorted true ratings (perfect ranking)
    sorted_ideal_ratings = np.sort(true_ratings)[::-1][:k]
    idcg = 0.0
    for i in range(len(sorted_ideal_ratings)):
        idcg += (2 ** sorted_ideal_ratings[i] - 1) / np.log2(i + 2)

    # Handle the case where IDCG is 0 (to avoid division by 0)
    if idcg == 0:
        return 0.0

    return dcg / idcg  # Normalized DCG


# Main logic to run UserKNN

dataset = '1m'

# Load example data
if dataset == '100k':
    user_item_matrix = MD.load_user_item_matrix_100k()
    gender = MD.load_gender_vector_100k()
    X = MD.load_user_item_matrix_100k()
    user_gender = MD.gender_user_dictionary_100k()
    gen_pref = pd.read_csv('ml-100k/user_genre_matrix_round.dat', delimiter=',')
    gen_pref = gen_pref.drop(columns=['userid'])  # -- to execute the code, manually set userid in the file as header
    output_file = 'ml-100k/Dist/'

elif dataset == '1m':
    user_item_matrix = MD.load_user_item_matrix_1m()
    gender = MD.load_gender_vector_1m()
    X = MD.load_user_item_matrix_1m()
    user_gender = MD.gender_user_dictionary_1m()
    gen_pref = pd.read_csv('ml-1m/user_genre_matrix_round.dat', delimiter=',')
    gen_pref = gen_pref.drop(columns=['userid'])  # -- to execute the code, manually set userid in the file as header
    output_file = 'ml-1m/Dist/'

user_item_matrix = csr_matrix(user_item_matrix)

# Split the data
train_matrix, test_matrix = split_train_test(user_item_matrix)
male_indices, female_indices, TMM, TMF = split_testmatrix_by_gender(test_matrix, user_gender)

# Step 1: Compute user similarity matrix
#similarity_matrix = compute_user_similarity(train_matrix)
rating_similarity_matrix = compute_user_similarity(train_matrix)
genre_similarity_matrix = compute_user_similarity(gen_pref)
similarity_matrix = compute_combine_sim(rating_similarity_matrix, genre_similarity_matrix)

# Step 2: Evaluate the UserKNN model
k_neighbors = 10
hit_rate, true_rate, predicted_rate, nDCG = evaluate_userknn(train_matrix, test_matrix, similarity_matrix, k=k_neighbors)
mae = calculate_mae(true_rate, predicted_rate)
average_ndcg = np.mean(nDCG)

print(f"HitRate@{k_neighbors}: {hit_rate:.4f}")
print(f'nDCG: {average_ndcg:.4f}')
print(f'MAE: {mae:.4f}')
