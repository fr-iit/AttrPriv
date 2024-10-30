import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import SplitData as SD
import RecSys_DataLoader as DL
import RecSys_Evaluation as RSE


class BPRMF:
    def __init__(self, num_users, num_items, num_factors, learning_rate, reg, num_epochs):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_epochs = num_epochs

        # Initialize the user and item latent factor matrices
        self.user_factors = np.random.normal(0, 0.01, (num_users, num_factors))
        self.item_factors = np.random.normal(0, 0.01, (num_items, num_factors))

    def fit(self, train_matrix):
        # Train the model using stochastic gradient descent (SGD)
        for epoch in range(self.num_epochs):
            for user in range(self.num_users):

                positive_items = train_matrix[user].nonzero()[1]
                for i in positive_items:
                    j = np.random.randint(self.num_items)
                    while train_matrix[user, j] > 0:
                        j = np.random.randint(self.num_items)
                    self.update_factors(user, i, j)
            print(f"Epoch {epoch + 1}/{self.num_epochs} completed")

    def update_factors(self, user, i, j):
        # Update latent factors for a single user and pair of items (i, j)
        u_factors = self.user_factors[user]
        i_factors = self.item_factors[i]
        j_factors = self.item_factors[j]


        x_uij = np.dot(u_factors, i_factors) - np.dot(u_factors, j_factors)
        sigmoid_x_uij = 1 / (1 + np.exp(x_uij))
        # print(f"sig: {sigmoid_x_uij}")

        # Gradient update
        self.user_factors[user] += self.learning_rate * (
                    (sigmoid_x_uij * (i_factors - j_factors)) - self.reg * u_factors)
        self.item_factors[i] += self.learning_rate * (sigmoid_x_uij * u_factors - self.reg * i_factors)
        self.item_factors[j] += self.learning_rate * (-sigmoid_x_uij * u_factors - self.reg * j_factors)

    def predict(self, user, item):
        # print(np.dot(self.user_factors[user], self.item_factors[item]))
        return np.dot(self.user_factors[user], self.item_factors[item])


# def bprmf_predict_func(model):
#     def predict(user_indices, item_indices):
#         # Use the model's predict function to get predicted ratings
#         return model.predict(user_indices, item_indices)
#     return predict

def bprmf_predict_func(model):
    def predict(user_indices, item_indices):
        # Predict ratings for each user-item pair individually
        return np.array([model.predict(u, i) for u, i in zip(user_indices, item_indices)])
    return predict

def evaluate_1_plus_random_HitRate(R_train, R_test, model, top_k=10):
    num_users, num_items = R_train.shape
    hit_count = 0
    num_tests = 0
    num_candidates = 1000

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

            scores = [model.predict(u, candidate) for candidate in candidates]
            top_k_items = np.argsort(scores)[-top_k:]
            # print(f"user {u}, top k: {top_k_items} ")

            if 0 in top_k_items:
                hit_count += 1

            num_tests += 1

    print(f"hit count: {hit_count} & test case no. : {num_tests}")
    hr_at_k = hit_count / num_tests if num_tests > 0 else 0

    return hr_at_k


def evaluate_multiple_times(train_matrix, test_matrix, model, num_runs, k):
    hr_results = []
    mae_results = []
    bprmf_predict = bprmf_predict_func(model)
    for _ in range(num_runs):

        mae = RSE.calculate_mae(test_matrix, bprmf_predict)
        mae_results.append(mae)
        hr_at_k = evaluate_1_plus_random_HitRate(train_matrix, test_matrix, model, top_k=k)
        hr_results.append(hr_at_k)

    avg_hr = np.mean(hr_results)
    std_hr = np.std(hr_results)

    print(f'MAE: {np.mean(mae_results): .2f}, std: {np.std(mae_results): .2f}')

    return avg_hr, std_hr


dataset = 'yahoo'

# Load example data
if dataset == '1m':
    user_item_matrix = DL.load_user_item_matrix_1m()
    # for ml1m: Best HR@10: 0.2070 with Factors: 30, Epochs: 30, Learning Rate: 0.01, Regularization: 0.001
    Factors = 30
    Epochs = 30
    Learning_Rate = 0.01
    Regularization = 0.001
elif dataset == '100k':
    user_item_matrix = DL.load_user_item_matrix_100k()
    # for ml100k: Best HR@10: 0.1709 with Factors: 30, Epochs: 30, Learning Rate: 0.1, Regularization: 0.01
    Factors = 30
    Epochs = 30
    Learning_Rate = 0.1
    Regularization = 0.01
elif dataset == 'yahoo':
    user_item_matrix = DL.load_user_item_matrix_yahoo()
    # for yahoo, Best HR@10: 0.5267 with Factors: 30, Epochs: 30, Learning Rate: 0.1, Regularization: 0.01
    Factors = 30
    Epochs = 30
    Learning_Rate = 0.1
    Regularization = 0.01

    print(dataset+' - loaded')


user_item_matrix = csr_matrix(user_item_matrix)
#print(user_item_matrix.shape)
# Split the data
train_matrix, test_matrix = SD.split_train_test(user_item_matrix)
print(f'train_matrix: {train_matrix.shape}, test_matrix: {test_matrix.shape}')


# # --- start Hyperparameter tuning using cross-validation
# best_hr = 0
# best_factors = None
# best_epochs = None
# best_lr = None
# best_reg = None
#
# for num_factors in [10, 20, 30]:  # Number of latent factors
#     for num_epochs in [10,20, 30]:  # Number of epochs ->
#         for learning_rate in [0.001, 0.01, 0.1]:  # Learning rate ->
#             for reg in [0.001, 0.01, 0.1]:  # Regularization term ->
#                 print(
#                     f"Testing num_factors={num_factors}, num_epochs={num_epochs}, learning_rate={learning_rate}, reg={reg}")
#                 bpr_model = BPRMF(num_users=train_matrix.shape[0], num_items=train_matrix.shape[1],
#                                   num_factors=num_factors, learning_rate=learning_rate,
#                                   reg=reg, num_epochs=num_epochs)
#                 bpr_model.fit(train_matrix)
#
#                 hr = evaluate_1_plus_random_HitRate(train_matrix, test_matrix, bpr_model, top_k=10)
#                 print(f"HR@10: {hr:.4f}")
#
#                 if hr > best_hr:
#                     best_hr = hr
#                     print(f"best_hr: {best_hr: .4f}")
#                     best_factors = num_factors
#                     best_epochs = num_epochs
#                     best_lr = learning_rate
#                     best_reg = reg
#
#                 print(f"Best HR@10: {best_hr:.4f} with Factors: {best_factors}, Epochs: {best_epochs}, Learning Rate: {best_lr}, Regularization: {best_reg}")
# #
# # Final evaluation with the best hyperparameters
# bpr_model = BPRMF(num_users=train_matrix.shape[0],
#                   num_items=train_matrix.shape[1],
#                   num_factors=best_factors,
#                   learning_rate=best_lr,
#                   reg=best_reg,
#                   num_epochs=best_epochs)
# bpr_model.fit(train_matrix)
# --- end hypertuning

bpr_model = BPRMF(num_users=train_matrix.shape[0], num_items=train_matrix.shape[1], num_factors=Factors, num_epochs=Epochs, learning_rate=Learning_Rate, reg=Regularization)
bpr_model.fit(train_matrix)
# bprmf_predict = bprmf_predict_func(bpr_model)

avg_hr, std_hr = evaluate_multiple_times(train_matrix, test_matrix, bpr_model, num_runs=5, k=10)

print(f"1+Random avg HitRatio@10: {avg_hr:.4f} & Std HR: {std_hr:.4f}")
