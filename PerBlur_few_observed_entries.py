import time
import RecSys_DataLoader as DL
import numpy as np
from six.moves import range
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def knn_impute_few_observed(X, missing_mask, k=30, verbose=False, print_interval=100):

    print("Starting KNN imputation...")
    start_t = time.time()
    n_rows, n_cols = X.shape
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major
    X_column_major = X.copy(order="F")

    from sklearn.metrics import pairwise_distances

    # Compute pairwise distances, using a small epsilon to avoid division by zero
    epsilon = 1e-10
    D = pairwise_distances(X, metric='euclidean') + epsilon

    # Use a large value to denote effectively infinite distances
    effective_infinity = np.max(D) * 2

    # This is a simple way to simulate X_row_major, D and effective_infinity
    X_row_major = X.copy()

    # Handle infinities and valid distances
    D_sorted = np.argsort(D, axis=1)
    inv_D = 1.0 / D
    D_valid_mask = D < effective_infinity
    valid_distances_per_row = D_valid_mask.sum(axis=1)

    D_sorted = [D_sorted[i, :count] for i, count in enumerate(valid_distances_per_row)]

    k_nearest_indices_filter = {}
    print("initial")

    for i in range(n_rows):
        missing_row = missing_mask[i, :]
        missing_indices = np.where(missing_row)[0]
        row_weights = inv_D[i, :]

        candidate_neighbor_indices = D_sorted[i]
        user_filter = []

        for j in missing_indices:
            observed = observed_mask_column_major[:, j]
            sorted_observed = observed[candidate_neighbor_indices]
            observed_neighbor_indices = candidate_neighbor_indices[sorted_observed]
            k_nearest_indices = observed_neighbor_indices[:k]

            user_filter.append(list(k_nearest_indices))

            k_nearest_indices_filter[str(i)] = user_filter.copy()

            weights = row_weights[k_nearest_indices]
            weight_sum = weights.sum()
            if weight_sum > 0:
                column = X_column_major[:, j]
                values = column[k_nearest_indices]
                X_row_major[i, j] = np.dot(values, weights) / weight_sum

    print("Imputation completed.")
    print("Saving results...")

    #Save the imputed data
    output_file = "ml-yahoo/"
    with open(output_file + "TrainingSet_users_KNN_fancy_imputation_k_30.dat", 'w') as f:
        for index_user, user in enumerate(X_row_major):
            for index_movie, rating in enumerate(user):
                if not np.isnan(rating):  # Check if rating is not NaN
                    f.write(f"{index_user + 1}::{index_movie + 1}::{int(np.round(rating))}::0000000\n")

    print("Save the k-nearest neighbors indices")

    #Save the k-nearest neighbors indices
    with open(output_file + "NN_All_AllUsers_Neighbors_Weight_K_30_item_choice.json", "w") as fp:
        json.dump(k_nearest_indices_filter, fp, cls=NpEncoder)

    return X_row_major



X = DL.load_user_item_matrix_yahoo() #MD.load_user_item_matrix_100k()
    #MD.load_user_item_matrix_1m_trainingSet()

missing_mask = (X == 0)
#np.isnan(X)
print("mask")
#print(missing_mask)

# Impute missing values
X_imputed = knn_impute_few_observed(X, missing_mask, k=30, verbose=True)

print("Data imputation and saving completed.")
