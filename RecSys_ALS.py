import numpy as np
from scipy.sparse import csr_matrix
import RecSys_DataLoader as DL
from sklearn.model_selection import train_test_split
import RecSys_Evaluation as RSE


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
        # Ensure the user_index is within the valid range for R_test
        if user_index < R_test.shape[0]:  # Check if the user exists in the test matrix
            if gender == "M" or gender == 'm':
                male_indices.append(user_index)
                R_test_male.append(R_test[user_index, :].toarray())
            elif gender == "F" or gender == 'f':
                female_indices.append(user_index)
                R_test_female.append(R_test[user_index, :].toarray())
        else:
            print(f"Warning: user_index {user_index} is out of range for the test matrix.")
    print(f'male: {len(R_test_male)}')
    print(f'female: {len(R_test_female)}')
    # Convert to numpy arrays
    R_test_male = csr_matrix(np.vstack(R_test_male))  # Convert to sparse matrix
    R_test_female = csr_matrix(np.vstack(R_test_female))  # Convert to sparse matrix

    return np.array(male_indices), np.array(female_indices), R_test_male, R_test_female


def als_explicit(R, num_factors=10, regularization=0.1, num_iterations=20):
    """
    ALS implementation for explicit ratings.
    """
    num_users, num_items = R.shape
    P = np.random.rand(num_users, num_factors)
    Q = np.random.rand(num_items, num_factors)

    for iteration in range(num_iterations):
        for u in range(num_users):
            rated_items = R[u, :].nonzero()[1]
            if len(rated_items) > 0:
                Q_rated = Q[rated_items, :]
                R_u = R[u, rated_items].toarray().flatten()
                A = Q_rated.T @ Q_rated + regularization * np.eye(num_factors)
                b = Q_rated.T @ R_u
                P[u, :] = np.linalg.solve(A, b)

        for i in range(num_items):
            rated_users = R[:, i].nonzero()[0]
            if len(rated_users) > 0:
                P_rated = P[rated_users, :]
                R_i = R[rated_users, i].toarray().flatten()
                A = P_rated.T @ P_rated + regularization * np.eye(num_factors)
                b = P_rated.T @ R_i
                Q[i, :] = np.linalg.solve(A, b)

        prediction = P @ Q.T
        error = np.sqrt(np.sum((R.toarray()[R.nonzero()] - prediction[R.nonzero()]) ** 2) / len(R.nonzero()[0]))
        print(f"Iteration {iteration + 1}/{num_iterations}, RMSE: {error:.4f}")

    return P, Q

def evaluate_1_plus_random_HitRate(R_train, R_test, P, Q, top_k=10):

    num_users = R_test.shape[0]#R_train.shape
    num_items = R_train.shape[1]
    hit_count = 0
    num_tests = 0
    num_candidates = int(num_items*0.2)
    print(num_candidates)

    #print(f"Number of users: {num_users} & Number of items: {num_items}")

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        for item in rated_items:
            non_rated_items = np.setdiff1d(np.arange(num_items), R_train[u, :].nonzero()[1])

            if len(non_rated_items) < num_candidates - 1:
                continue  # Not enough non-rated items to sample

            random_samples = np.random.choice(non_rated_items, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([item], random_samples))

            scores = P[u, :] @ Q[candidates, :].T
            top_k_items = np.argsort(scores)[-top_k:]
            #print(f"user {u}, top k: {top_k_items} ")

            if 0 in top_k_items:
                hit_count += 1

            num_tests += 1

    # print(f"hit count: {hit_count} & test case no. : {num_tests}")
    hr_at_k = hit_count / num_tests if num_tests > 0 else 0

    return hr_at_k


def measure_gender_stereotypical_recommendations(R_train, R_test, gender, P, Q, topM, topF, top_k=10):
    # Define gender-stereotypical items

    topM = topM #[751, 327, 24, 688, 472, 888, 317, 948, 315, 178]  # Male-preferred items
    topF = topF #[332, 321, 877, 906, 827, 882, 337, 278, 872, 304]  # Female-preferred items

    num_users = R_test.shape[0]  # R_train.shape
    num_items = R_train.shape[1]
    num_candidates = int(num_items*0.2) #1000
    print(num_candidates)
    total_recommended_items = 0
    male_stereotypical_count = 0  # To track recommendations matching top10M
    female_stereotypical_count = 0  # To track recommendations matching top10F

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        for item in rated_items:
            non_rated_items = np.setdiff1d(np.arange(num_items), R_train[u, :].nonzero()[1])

            if len(non_rated_items) < num_candidates - 1:
                continue  # Not enough non-rated items to sample

            random_samples = np.random.choice(non_rated_items, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([item], random_samples))

            scores = P[u, :] @ Q[candidates, :].T
            top_k_items = np.argsort(scores)[-top_k:]

        # Initialize hits to 0 to avoid uninitialized variable error
        male_hits = 0
        female_hits = 0

        if gender[u] == 0:
            male_hits = np.intersect1d(top_k_items, topM).size  # Count male-stereotypical hits
        elif gender[u] == 1:
            female_hits = np.intersect1d(top_k_items, topF).size  # Count female-stereotypical hits

        # Measure gender-stereotypical recommendations
        # male_hits = np.intersect1d(top_k_items, top10M).size  # Count male-stereotypical hits
        # female_hits = np.intersect1d(top_k_items, top10F).size  # Count female-stereotypical hits

        male_stereotypical_count += male_hits
        female_stereotypical_count += female_hits

        total_recommended_items += top_k

    # Calculate the proportion of gender-stereotypical items
    male_stereotypical_ratio = male_stereotypical_count / total_recommended_items
    female_stereotypical_ratio = female_stereotypical_count / total_recommended_items

    return male_stereotypical_ratio, female_stereotypical_ratio

def identify_popular_and_long_tail_items(X):

    # Step 1: Count the number of ratings per item (non-zero entries)
    item_rating_counts = np.count_nonzero(X, axis=0)

    # Step 2: Calculate the total number of ratings
    total_ratings = np.sum(item_rating_counts)

    # Step 3: Calculate cumulative percentage of ratings
    sorted_indices = np.argsort(item_rating_counts)[::-1]  # Sort items by count in descending order
    sorted_counts = item_rating_counts[sorted_indices]     # Get the sorted counts
    cumulative_counts = np.cumsum(sorted_counts)           # Calculate cumulative ratings

    # Step 4: Identify the split index for 80% of the total ratings
    split_index = np.searchsorted(cumulative_counts, 0.8 * total_ratings)

    # Step 5: Classify items as popular and long-tail
    popular_items = sorted_indices[:split_index + 1]  # Items contributing to 80% of total ratings
    long_tail_items = sorted_indices[split_index + 1:]  # Remaining items (20%)

    return popular_items, long_tail_items

# --- results are better with 1+random
def measure_popular_long_tail_recommendations(R_train, R_test, P, Q, original_data, top_k=10):

    popular_items, long_tail_items = identify_popular_and_long_tail_items(original_data)
    print(f'popular_items: {len(popular_items)}, long_tail_items: {len(long_tail_items)} ')

    num_users = R_test.shape[0]
    num_items = R_train.shape[1]
    total_recommended_items = 0
    popular_count = 0  # To track recommendations matching popular items
    long_tail_count = 0  # To track recommendations matching long-tail items
    num_candidates = int(num_items*0.2)#1000
    print(num_candidates)

    for u in range(num_users):
        rated_items = R_test[u, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        for item in rated_items:
            non_rated_items = np.setdiff1d(np.arange(num_items), R_train[u, :].nonzero()[1])

            if len(non_rated_items) < num_candidates - 1:
                random_samples = non_rated_items  # Use all non-rated items if fewer than needed
            else:
                random_samples = np.random.choice(non_rated_items, size=num_candidates - 1, replace=False)

            # Combine the test item with sampled non-rated items to form candidates
            candidates = np.concatenate(([item], random_samples))

            # Compute recommendation scores for candidates
            scores = P[u, :] @ Q[candidates, :].T
            #top_k_items = candidates[np.argsort(scores)[-top_k:][::-1]]  # Sorting in descending order
            top_k_items = np.argsort(scores)[-top_k:]

            # Measure popular and long-tail recommendations
            popular_hits = np.intersect1d(top_k_items, popular_items).size  # Count popular item hits
            long_tail_hits = np.intersect1d(top_k_items, long_tail_items).size  # Count long-tail item hits

            # Accumulate the counts
            popular_count += popular_hits
            long_tail_count += long_tail_hits
            total_recommended_items += top_k

    # Calculate the proportion of popular and long-tail items
    popular_ratio = popular_count / total_recommended_items if total_recommended_items > 0 else 0
    long_tail_ratio = long_tail_count / total_recommended_items if total_recommended_items > 0 else 0

    return popular_ratio, long_tail_ratio

def als_predict_func(P, Q):
    def predict(user_indices, item_indices):
        # Use the dot product of P (user factors) and Q (item factors) to predict ratings
        return np.array([P[u, :] @ Q[i, :].T for u, i in zip(user_indices, item_indices)])
    return predict


def evaluate_multiple_times(train_matrix, test_matrix, TMM, TMF, male_indices, female_indices, P, Q, original_data, als_predict, topM, topF, num_runs, k):
    mae_results = []
    hr_results = []
    TMM_result = []
    TMF_result = []
    pop_re = []
    abg_pop_re = []
    agg_div = []

    # divrsity : 1) male vs female in topN and 2) popular vs long tail
    div_M = []
    div_F = []
    div_popular = []
    div_long = []
    nDCG_result = []

    i = 1
    for _ in range(num_runs):
        print(f"i: {i}")
        # mae = RSE.calculate_mae(test_matrix, als_predict)
        #
        # nDCG = RSE.evaluate_1_plus_random_nDCG(train_matrix, test_matrix, P, Q, k)
        # nDCG_result.append(nDCG)
        #
        # hr_at_k = evaluate_1_plus_random_HitRate(train_matrix, test_matrix, P, Q, k)
        #     #evaluate_hit_rate_and_popularity(train_matrix, test_matrix, P, Q, top_k=k)
        #     #evaluate_1_plus_random_HitRate(train_matrix, test_matrix, P, Q, top_k=k)
        #
        # _, pop, avg_pop, aggdiv = RSE.evaluate_hit_rate_and_popularity(train_matrix, test_matrix, P, Q, k)
        #
        # mae_results.append(mae)
        # hr_results.append(hr_at_k)
        # pop_re.append(pop)
        # abg_pop_re.append(avg_pop)
        # agg_div.append(aggdiv)
        #
        # hr_k_F = evaluate_1_plus_random_HitRate(train_matrix, TMF, P[female_indices, :], Q, 10)
        # hr_k_M = evaluate_1_plus_random_HitRate(train_matrix, TMM, P[male_indices, :], Q, 10)

        # hr_k_F, _, _, _ = RSE.evaluate_hit_rate_and_popularity(train_matrix, TMF, P[female_indices, :], Q, k)
        # hr_k_M, _, _, _ = RSE.evaluate_hit_rate_and_popularity(train_matrix, TMM, P[male_indices, :], Q, k)

        # TMM_result.append(hr_k_M)
        # TMF_result.append(hr_k_F)

        male_ratio, female_ratio = measure_gender_stereotypical_recommendations(train_matrix, test_matrix, gender, P, Q, topM, topF)
        div_M.append(male_ratio)
        div_F.append(female_ratio)

        # Example Usage
        popular_ratio, long_tail_ratio = measure_popular_long_tail_recommendations(train_matrix, test_matrix, P, Q, original_data)
        div_popular.append(popular_ratio)
        div_long.append(long_tail_ratio)

        i= i+1

    # print("--- Performance ---")
    # print(f'Avg. MAE: {np.mean(mae_results): .4f}, std.: {np.std(mae_results): .4f} ')
    # print('(1+random protocol)')
    # print(f'Avg. HR: {np.mean(hr_results): .4f}, std.: {np.std(hr_results): .4f}')
    # print(f"Avg. nDCG: {np.mean(nDCG_result): .4f}, .std: {np.std(nDCG_result): .4f}")
    # print("--- Gender Bias ---")
    # print(f"Male -> avg HR: {np.mean(TMM_result): .4f} & std: {np.std(TMM_result): .4f}")
    # print(f"Female -> avg HR: {np.mean(TMF_result): .4f} & std: {np.std(TMF_result): .4f}")
    # print("--- Popularity Bias ---")
    # print(f'popular : {np.mean(pop_re): .4f}, avg popular: {np.mean(abg_pop_re): .4f}')
    # print(f'aggregate diversity: {np.mean(agg_div): .4f}')
    print("--- Diversity (1+random protocol) ---")
    print(f"Male-stereotypical ratio: {np.mean(div_M):.4f}")
    print(f"Female-stereotypical ratio: {np.mean(div_F):.4f}")
    print(f"Popular item ratio: {np.mean(div_popular):.4f}")
    print(f"Long-tail item ratio: {np.mean(div_long):.4f}")


# ------- call function and load data
dataset = 'yahoo'

# Load example data
if dataset == '100k':
    user_item_matrix = DL.load_user_item_matrix_100k() # DL.load_user_item_matrix_100k_masked() #
    gender = DL.load_gender_vector_100k()
    X = DL.load_user_item_matrix_100k()
    user_gender = DL.gender_user_dictionary_100k()
    top10M = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178]
    top10F = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304]
elif dataset == '1m':
    user_item_matrix = DL.load_user_item_matrix_1m() # DL.load_user_item_matrix_1m_masked() #
    gender = DL.load_gender_vector_1m()
    X = DL.load_user_item_matrix_1m()
    user_gender = DL.gender_user_dictionary_1m()
    # Factors = 30
    # Iterations = 30
elif dataset == 'yahoo':
    user_item_matrix = DL.load_user_item_matrix_yahoo()
    gender = DL.load_gender_vector_yahoo()
    X = DL.load_user_item_matrix_yahoo()
    user_gender = DL.gender_user_dictionary_yahoo()
    # top10M = [8901, 797, 8045, 8883, 272, 6292, 7715, 8611, 8409, 9107]
    # top10F = [5925, 4959, 8305, 8607, 9276, 5791, 730, 1155, 34, 1743]
    top10M = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178]
    top10F = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304]

user_item_matrix = csr_matrix(user_item_matrix)
# Split the data
train_matrix, test_matrix = split_train_test(user_item_matrix)#SD.split_train_test(user_item_matrix)
male_indices, female_indices, TMM, TMF = split_testmatrix_by_gender(test_matrix, user_gender)
print(dataset)
Factors = 30
Iterations = 30

# Hyperparameter tuning (cross-validation)
# best_hr = 0
# best_factors = 0
# best_iterations = 0
# best_reg = 0
# for num_factors in [10, 20, 30]:
#     for num_iterations in [20, 25, 30]:
#          for reg in [0.1]: # most of time reg -> 0.1 shows best result
#             print(f"factor: {num_factors}, iteration: {num_iterations}, reg: {reg}")
#             P, Q = als_explicit(train_matrix, num_factors=num_factors, regularization=reg, num_iterations=num_iterations)
#             hr = evaluate_1_plus_random_HitRate(train_matrix, test_matrix, P, Q, top_k=10)
#             print(f"Factors: {num_factors}, Iterations: {num_iterations}, Reg: {reg} HR@10: {hr:.4f}")
#
#             if hr > best_hr:
#                 best_hr = hr
#                 best_factors = num_factors
#                 best_iterations = num_iterations
#                 best_reg = reg
#
#             print(f"Best HR@10: {best_hr:.4f} with Factors: {best_factors}, reg: {best_reg} and Iterations: {best_iterations}")
#
#
# # Final evaluation with the best hyperparameters
# P, Q = als_explicit(train_matrix, num_factors=best_factors, regularization=best_reg, num_iterations=best_iterations)
# print('finish')
# print(f"Best HR@10: {best_hr:.4f} with Factors: {best_factors}, reg: {best_reg} and Iterations: {best_iterations}")
# --- end hyperparameter

# Best HR@10: 0.0086 with Factors: 30 and Iterations: 30 -> ml1m
# Best HR@10: 0.0182 with Factors: 30, reg: 0.1 and Iterations: 30 -> ml100k
# Best HR@10: 0.0728 with Factors: 30, reg: 0.1 and Iterations: 30 -> ml-yahoo
P, Q = als_explicit(train_matrix, num_factors=Factors, regularization=0.1, num_iterations=Iterations)
als_predict = als_predict_func(P, Q)

evaluate_multiple_times(train_matrix, test_matrix, TMM, TMF, male_indices, female_indices, P, Q, X, als_predict, top10M, top10F, num_runs = 5 , k= 10)
# nDCG = RSE.evaluate_1_plus_random_nDCG(train_matrix, test_matrix, P, Q)
