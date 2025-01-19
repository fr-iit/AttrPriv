import numpy as np

def mae(predictions, test_matrix):

    # Extract non-zero entries from the test matrix
    test_nonzero_indices = test_matrix.nonzero()
    test_values = test_matrix[test_nonzero_indices]

    max_index = np.max(test_nonzero_indices)
    #print(f"Max index in test_nonzero_indices: {max_index}")

    # Predict ratings for the non-zero entries
    predicted_values = predictions[test_nonzero_indices]

    # Calculate the mean absolute error
    mae = np.mean(np.abs(test_values - predicted_values))

    return mae

def calculate_mae(test_matrix, predict_func):

    row_indices, col_indices = test_matrix.nonzero()
    actual_ratings = test_matrix[row_indices, col_indices].A1  # Flatten the array
    predicted_ratings = predict_func(row_indices, col_indices)
    mae = np.mean(np.abs(actual_ratings - predicted_ratings))
    return mae

def calculate_normalized_mae(R_test, predictions, min_rating = 1, max_rating = 5):

    # Extract non-zero test ratings and corresponding predictions
    test_indices = R_test.nonzero()
    true_ratings = R_test[test_indices]
    predicted_ratings = predictions[test_indices]

    # Calculate MAE
    mae = np.mean(np.abs(true_ratings - predicted_ratings))

    # Normalize MAE
    nmae = mae / (max_rating - min_rating)

    return nmae


def calculate_rmse(test_matrix, P, Q):

    # Check if test_matrix dimensions match P and Q
    num_users, num_items = test_matrix.shape
    if num_users > P.shape[0] or num_items > Q.shape[0]:
        test_matrix = test_matrix[:P.shape[0], :Q.shape[0]]

    # Compute the predicted ratings
    predictions = P @ Q.T  # predictions is a NumPy array with shape (6040, 3952)

    # Extract non-zero entries from the test matrix
    row_indices, col_indices = test_matrix.nonzero()

    max_row_index = np.max(row_indices)
    max_col_index = np.max(col_indices)

    # Ensure indices are within bounds
    assert max_row_index < predictions.shape[0], "Row index out of bounds"
    assert max_col_index < predictions.shape[1], "Column index out of bounds"

    # Get the actual test ratings
    test_values = test_matrix[row_indices, col_indices].A1  # Convert to 1D array using .A1

    # Predict ratings for the corresponding user-item pairs
    predicted_values = predictions[row_indices, col_indices]

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(np.mean((test_values - predicted_values) ** 2))

    return rmse

def calculate_nDCG_at_k(scores, relevant_index, k=10):

    ranking = np.argsort(scores)[::-1][:k]

    # Check if the relevant item is within the top-k
    if relevant_index not in ranking:
        return 0.0

    # Position of the relevant item in the top-k ranking
    rank_position = np.where(ranking == relevant_index)[0][0]

    # Calculate DCG
    dcg = 1 / np.log2(rank_position + 2)

    # Ideal DCG (IDCG) is 1 because the ideal rank for the relevant item is the top position
    idcg = 1.0

    # Calculate nDCG
    nDCG = dcg / idcg
    return nDCG


def evaluate_1_plus_random_nDCG(R_train, R_test, P, Q, top_k):

    num_users, num_items = R_train.shape
    num_candidates = int(num_items*0.2) #1000
    nDCG_scores = []

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        # Iterate over all rated items in the test set
        for item in rated_items:
            # Generate random negative samples
            non_rated_items = np.setdiff1d(np.arange(num_items), R_train[u, :].nonzero()[1])

            if len(non_rated_items) < num_candidates - 1:
                continue  # Not enough non-rated items to sample

            random_samples = np.random.choice(non_rated_items, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([item], random_samples))

            # Predict scores for the selected items
            scores = P[u, :] @ Q[candidates, :].T

            # Calculate nDCG@k
            nDCG_at_k = calculate_nDCG_at_k(scores, relevant_index=0, k=top_k)

            nDCG_scores.append(nDCG_at_k)

    avg_nDCG_at_k = np.mean(nDCG_scores) if nDCG_scores else 0

    return avg_nDCG_at_k

def calculate_hit_ratio(R, predicted_R, top_k):
 
    hit_count = 0
    num_users = R.shape[0]

    for i in range(num_users):
        true_items = set(np.nonzero(R[i, :])[0])
        if len(true_items) == 0:
            continue

        top_k_items = np.argsort(predicted_R[i, :])[-top_k:]
        if any(item in true_items for item in top_k_items):
            hit_count += 1

    hit_ratio_at_k = hit_count / num_users
    return hit_ratio_at_k

def evaluate_hit_rate_and_popularity(R_train, R_test, P, Q, top_k=10):
 
    num_users = R_test.shape[0]  # R_train.shape
    num_items = R_train.shape[1]
    hit_count = 0
    total_recommended_items = 0

    # Calculate item popularity
    item_popularity = np.array(R_train.sum(axis=0)).flatten()

    total_popularity = 0
    unique_recommended_items = set()  # Track unique recommended items

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        # Predict scores for all items
        scores = P[u, :] @ Q.T
        top_k_items = np.argsort(scores)[-top_k:]

        # Update the set of unique recommended items
        unique_recommended_items.update(top_k_items)

        # Check for hits
        hits = np.intersect1d(rated_items, top_k_items)
        hit_count += len(hits)

        # Accumulate popularity of the top-k recommended items
        total_popularity += np.sum(item_popularity[top_k_items])
        total_recommended_items += top_k

    hr_at_k = hit_count / num_users if num_users > 0 else 0

    avg_popularity = total_popularity / total_recommended_items if total_recommended_items > 0 else 0
    avg_item_popularity = np.mean(item_popularity[item_popularity > 0])
    avg_popularity_diff = avg_popularity - avg_item_popularity

    # Calculate aggregate diversity as the size of the unique items set
    aggregate_diversity = len(unique_recommended_items)
    #print(aggregate_diversity)

    norm_agg_div = aggregate_diversity/num_items

    return hr_at_k, avg_popularity, avg_popularity_diff, norm_agg_div


# popular & long tail item ratio
def categorize_items_by_popularity(item_popularity, popular_threshold=0.8):
    
    popularity_threshold = np.percentile(item_popularity[item_popularity > 0], 100 * (1 - popular_threshold))
    popular_items = set(np.where(item_popularity >= popularity_threshold)[0])
    long_tail_items = set(np.where(item_popularity < popularity_threshold)[0])
    print(f"PI: {len(popular_items)} & LI: {len(long_tail_items)}")
    return popular_items, long_tail_items


def evaluate_hit_rate_and_popularity_with_distribution(R_train, R_test, P, Q, top_k=10):
    
    num_users, num_items = R_train.shape
    hit_count = 0
    total_recommended_items = 0
    popular_threshold = 0.8

    # Calculate item popularity
    item_popularity = np.array(R_train.sum(axis=0)).flatten()
    popular_items, long_tail_items = categorize_items_by_popularity(item_popularity, popular_threshold)

    total_popularity = 0
    popular_count = 0
    long_tail_count = 0

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        # Predict scores for all items
        scores = P[u, :] @ Q.T
        top_k_items = np.argsort(scores)[-top_k:]

        # Check for hits
        hits = np.intersect1d(rated_items, top_k_items)
        hit_count += len(hits)

        # Accumulate popularity of the top-k recommended items
        total_popularity += np.sum(item_popularity[top_k_items])
        total_recommended_items += top_k

        # Count how many of the top-k items are popular vs long-tail
        popular_count += len([item for item in top_k_items if item in popular_items])
        long_tail_count += len([item for item in top_k_items if item in long_tail_items])

    hr_at_k = hit_count / num_users if num_users > 0 else 0
    avg_popularity = total_popularity / total_recommended_items if total_recommended_items > 0 else 0
    avg_item_popularity = np.mean(item_popularity[item_popularity > 0])
    avg_popularity_diff = avg_popularity - avg_item_popularity

    popular_ratio = popular_count / total_recommended_items if total_recommended_items > 0 else 0
    long_tail_ratio = long_tail_count / total_recommended_items if total_recommended_items > 0 else 0

    return hr_at_k, avg_popularity, avg_popularity_diff

