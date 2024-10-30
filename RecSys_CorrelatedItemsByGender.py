import MovieLensData as MD
import RecSys_DataLoader as DL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def find_Corelated_items_by_gender(data_version = 'yahoo'):

    # Load user-item matrix and gender vector based on the data version
    if data_version == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        rank = 2000
    elif data_version == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        rank = 1000
    elif data_version == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        rank = 1000

    print("---- Data Loaded Successfully ----")

    print("----------------- start ------")

    # Split the data into training and test sets
    split_point = int(0.8 * len(X))
    X_train, T_train = X[:split_point], T[:split_point]
    X_test, T_test = X[split_point:], T[split_point:]

    print("lists")

    # Initialize StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=10)
    avg_coefs = np.zeros(shape=(X_train.shape[1],))

    random_state = np.random.RandomState(0)
    coef_list = []

    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        model = LogisticRegression(penalty='l2', random_state=random_state, max_iter=1000)
        model.fit(x, t)

        # Record coefficients for each fold
        # ranks = ss.rankdata(model.coef_[0])
        # coefs.append(ranks)
        avg_coefs += model.coef_[0]
        coef_list.append(model.coef_[0])

    # Calculate the average coefficients
    avg_coefs /= cv.get_n_splits()

    # print(avg_coefs)

    # Average the ranks across folds
    # coefs = np.average(coefs, axis=0)
    # coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    # coefs = np.asarray(sorted(coefs, key=lambda x: x[2], reverse=True))  # Sort by coefficient value
    # #coefs = np.asarray(sorted(coefs))  # Sort by coefficient value

    # Rank the items based on average coefficients
    ranked_items = [[i, coef] for i, coef in enumerate(avg_coefs)]
    ranked_items = sorted(ranked_items, key=lambda x: x[1], reverse=True)

    # Create a list of tuples for female and male items along with their coefficients
    top_female_items_with_coef = [(item + 1, coef) for item, coef in ranked_items if coef > 0.01][:rank]
    top_male_items_with_coef = [(item + 1, coef) for item, coef in ranked_items if coef < -0.01][:rank]


    # Adjust the assignment of indicative items based on the coefficients
    # Since positive coefficients correspond to female users and negative correspond to male:
    # gender is label as male: 0 and female: 1
    # consider 1 based indexing
    # --- female
    top_female_items = [item for item, coef in top_female_items_with_coef]
    # --- male
    # Step 2: Sort the filtered list by coefficients in ascending order (more negative comes first)
    top_male_items_with_coef = sorted(top_male_items_with_coef, key=lambda x: x[1])
    top_male_items = [item for item, coef in top_male_items_with_coef]


    # top_female_items = [item + 1 for item, coef in ranked_items if coef > 0][:rank]  # Female items
    # top_male_items = [item + 1 for item, coef in ranked_items if coef < 0][:rank]  # Male items

    print(f"Top Male Correlated Items: {len(top_male_items)}")
    #print(top_male_items)
    print(f"Top Female Correlated Items: {len(top_female_items)}")
    #print(top_female_items)

    # Save results to files
    np.savetxt(f"ml-{data_version}/Other/ml{data_version}_Lm_Item.dat", top_male_items, delimiter=", ", fmt="%s")
    np.savetxt(f"ml-{data_version}/Other/ml{data_version}_Lf_Item.dat", top_female_items, delimiter=", ", fmt="%s")

    # Find items that are not in either the male or female list
    all_items = set(range(1, X_train.shape[1] + 1))  # Full set of items (1-based)
    male_female_items = set(top_male_items).union(set(top_female_items))  # Items in male or female list

    # Items not in male or female list
    non_correlated_items = all_items - male_female_items

    # Ensure `top_male_items` are added first, then non-correlated items
    # 1. Convert `top_male_items` and `non_correlated_items` to lists (to maintain order)
    ordered_male_item_list = top_male_items + list(non_correlated_items)
    # print(f"len: {len(ordered_male_item_list)} male")
    # print(ordered_male_item_list)

    # 2. Create the `male_item` set using the ordered list
    male_item = set(ordered_male_item_list)  # Create set from the ordered list

    # Do the same for female items, ensuring `top_female_items` appear first
    ordered_female_item_list = top_female_items + list(non_correlated_items)
    # print(f"len: {len(ordered_female_item_list)} female")
    # print(ordered_female_item_list)
    female_item = set(ordered_female_item_list)

    # Find common items
    c = set(ordered_male_item_list) & set(ordered_female_item_list)

    # Print the results
    #print(len(c))
    print(f"Male: {len(male_item)} & Female: {len(female_item)}")

    # Save results to files
    np.savetxt(f"ml-{data_version}/Other/common_ml{data_version}_Lm_Item.dat", ordered_male_item_list,
               delimiter=", ", fmt="%s")
    np.savetxt(f"ml-{data_version}/Other/common_ml{data_version}_Lf_Item.dat", ordered_female_item_list,
               delimiter=", ", fmt="%s")


    # Convert to numpy array to save in a text file
    female_array = np.array(top_female_items_with_coef, dtype=[('ItemID', 'i4'), ('Coefficient', 'f4')])
    male_array = np.array(top_male_items_with_coef, dtype=[('ItemID', 'i4'), ('Coefficient', 'f4')])

    # Save to file with headers
    np.savetxt(f"ml-{data_version}/Other/coef_ml{data_version}_Lm_Item_with_coef.dat", male_array, fmt="%d,%.4f",
               header="ItemID,Coefficient", comments='')
    np.savetxt(f"ml-{data_version}/Other/coef_ml{data_version}_Lf_Item_with_coef.dat", female_array, fmt="%d,%.4f",
               header="ItemID,Coefficient", comments='')

    # ----- check the coefficient with gender lebel

    # After fitting the logistic regression model, analyze the coefficients.
    # print("Analyzing Coefficients for Gender Mapping")
    #
    # # Example: Output the coefficients for the top items and check the signs
    # item_coefficients = model.coef_[0]  # Get the coefficients for the logistic regression model
    # print("Logistic Regression Coefficients (first 10 items):")
    # print(item_coefficients[:10])  # Print the first 10 coefficients for analysis
    #
    # # Check the mean rating values for male and female users separately
    # print("Average ratings for male and female users for each item:")
    #
    # # Get indices of male and female users from the gender vector
    # male_indices = np.where(T_train == 0)[0]  # Indices where gender is 0 (male)
    # female_indices = np.where(T_train == 1)[0]  # Indices where gender is 1 (female)
    #
    # # Calculate the average rating for male and female users separately for each item
    # avg_ratings_male = np.mean(X_train[male_indices], axis=0)  # Mean rating for male users
    # avg_ratings_female = np.mean(X_train[female_indices], axis=0)  # Mean rating for female users
    #
    # # Print the average rating for the first 10 items for each gender
    # print("Male Average Ratings (first 10 items):", avg_ratings_male[:10])
    # print("Female Average Ratings (first 10 items):", avg_ratings_female[:10])

    # ----- end checking

    # Convert to DataFrame for easier handling
    # coef_df = pd.DataFrame(coef_list).T
    # coef_df.columns = [f'Fold_{i+1}' for i in range(cv.get_n_splits())]
    # coef_df['ItemID'] = range(1, coef_df.shape[0] + 1)
    #
    # # Print a snippet of the coefficient data
    # print(coef_df.head())
    #
    # # Visualization: Boxplot to show the distribution of coefficients for each item
    # plt.figure(figsize=(14, 6))
    # coef_df.drop('ItemID', axis=1).boxplot()
    # plt.title('Stability of Logistic Regression Coefficients Across 10 Folds')
    # plt.xlabel('Cross-Validation Folds')
    # plt.ylabel('Coefficient Value')
    # plt.show()

    # # Find top correlated items
    # if top == -1:
    #     values = coefs[:, 2]
    #     var_val = np.min(np.abs(values))
    #     # index_zero = np.where(np.abs(values) == np.min(np.abs(values)))[0]
    #     # top_male = index_zero[0] if len(index_zero) > 0 else 0
    #     # top_female = index_zero[-1] if len(index_zero) > 0 else 0
    #     index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
    #     top_male = index_zero[0][0]
    #     top_female = index_zero[0][-1]
    #
    #     print(f"top_male: {top_male}, female: {top_female}")
    #
    #     L_m = coefs[:top_male, 1].astype(int)[:rank]
    #     R_m = 885 - coefs[:top_male, 0]  # Update with correct max item index
    #     C_m = np.abs(coefs[:top_male, 2])
    #     L_f = coefs[-top_female:, 1].astype(int)[:rank]  # Correct slicing for female items
    #     #L_f = coefs[coefs.shape[0] - top_female:, 1].astype(int)
    #     L_f = list(reversed(L_f))
    #     R_f = coefs[-top_female:, 0]
    #     R_f = list(reversed(R_f))
    #     C_f = coefs[-top_female:, 2]
    #     C_f = list(reversed(np.abs(C_f)))
    # else:
    #     L_m = coefs[:top, 1]
    #     R_m = 885 - coefs[:top, 0]  # Update with correct max item index
    #     C_m = np.abs(coefs[:top, 2])
    #     L_f = coefs[-top:, 1]
    #     L_f = list(reversed(L_f))
    #     R_f = coefs[-top:, 0]
    #     R_f = list(reversed(R_f))
    #     C_f = coefs[-top:, 2]
    #     C_f = list(reversed(np.abs(C_f)))

    # Find common items between male and female lists
    # common = [item for item in L_f if item in L_m]
    # print(f"common: {len(common)}")
    # print(len(L_f))

    # # Save the results to .dat files
    # np.savetxt("ml-1m/Other/ml100k_Lm_Item.dat", L_m, delimiter=", ", fmt="%s")
    # np.savetxt("ml-1m/Other/ml100k_Lf_Item.dat", L_f, delimiter=", ", fmt="%s")

def find_genre_of_correlatedItems():

    import pandas as pd

    column_names1 = ['MovieID']  # Specify the column names for the first file
    column_names2 = ['MovieID', 'Title', 'Genres']  # Specify the column names for the second file

    df_male = pd.read_csv('ml-1m/Other/ml100k_Lm_Item.dat', sep='::',header=None, names=column_names1, engine='python')  # Adjust 'sep' if using a different delimiter
    movie = pd.read_csv('ml-1m/movies.dat', sep=',',header=None, names=column_names2, engine='python')

    df_female = pd.read_csv('ml-1m/Other/ml100k_Lf_Item.dat', sep='::', header=None, names=column_names1,
                      engine='python')  # Adjust 'sep' if using a different delimiter

    df_male = df_male.astype(object)
    df_female = df_female.astype(object)

    # Step 2: Merge the DataFrames on the common column, e.g., 'MovieID'
    merged_df = pd.merge(df_male, movie[['MovieID', 'Genres']], on='MovieID')  # Use 'how' parameter if needed ('inner', 'outer', 'left', 'right')
    merged_female = pd.merge(df_female, movie[['MovieID', 'Genres']], on = 'MovieID')

    # Step 3: Save the merged DataFrame to a new .dat file
    merged_df.to_csv('ml-1m/Other/merge_ml100k_Lm_Items.dat', sep=':',  index=False, header=False)
    merged_female.to_csv('ml-1m/Other/merge_ml100k_Lf_Items.dat', sep=':',  index=False, header=False)


def find_Corelated_items_by_gender_rf(data_version='yahoo'):
    # Load user-item matrix and gender vector based on the data version
    if data_version == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        rank = 2000
    elif data_version == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        rank = 1000
    elif data_version == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        rank = 1000

    print("---- Data Loaded Successfully ----")

    # Split the data into training and test sets
    split_point = int(0.8 * len(X))
    X_train, T_train = X[:split_point], T[:split_point]
    X_test, T_test = X[split_point:], T[split_point:]

    print("Training data prepared")

    # Initialize StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=10)
    avg_importances = np.zeros(shape=(X_train.shape[1],))

    # Train Random Forest Classifier and collect feature importances
    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(x, t)

        # Collect feature importances for each fold
        avg_importances += rf_model.feature_importances_

    # Calculate the average importance across folds
    avg_importances /= cv.get_n_splits()

    # Create a list of tuples for item indices and their average importances
    ranked_items = [[i, importance] for i, importance in enumerate(avg_importances)]
    ranked_items = sorted(ranked_items, key=lambda x: x[1], reverse=True)  # Sort by descending importance

    # Separate items into female and male indicative lists based on feature importance
    # Use positive importances for female items and low importances for male
    print(f"median avg: {np.median(avg_importances)}")
    top_female_items_with_importance = [(item + 1, imp) for item, imp in ranked_items if imp > np.median(avg_importances)][:rank]  # Top female items
    top_male_items_with_importance = [(item + 1, imp) for item, imp in ranked_items if imp <= np.median(avg_importances)][:rank]  # Top male items

    # Extract item IDs only
    top_female_items = [item for item, imp in top_female_items_with_importance]
    top_male_items = [item for item, imp in top_male_items_with_importance]

    print(f"Top Male Correlated Items: {len(top_male_items)}")
    print(f"Top Female Correlated Items: {len(top_female_items)}")

    # Save results to files
    np.savetxt(f"ml-{data_version}/Other/ml{data_version}_RF_Lm_Item.dat", top_male_items, delimiter=", ", fmt="%s")
    np.savetxt(f"ml-{data_version}/Other/ml{data_version}_RF_Lf_Item.dat", top_female_items, delimiter=", ", fmt="%s")

    # Convert to numpy array to save in a text file
    female_array = np.array(top_female_items_with_importance, dtype=[('ItemID', 'i4'), ('Importance', 'f4')])
    male_array = np.array(top_male_items_with_importance, dtype=[('ItemID', 'i4'), ('Importance', 'f4')])

    # Save to file with headers
    np.savetxt(f"ml-{data_version}/Other/coef_ml{data_version}_RF_Lm_Item_with_importance.dat", male_array, fmt="%d,%.4f",
               header="ItemID,Importance", comments='')
    np.savetxt(f"ml-{data_version}/Other/coef_ml{data_version}_RF_Lf_Item_with_importance.dat", female_array, fmt="%d,%.4f",
               header="ItemID,Importance", comments='')

    # Find items that are not in either the male or female list
    all_items = set(range(1, X_train.shape[1] + 1))  # Full set of items (1-based)
    male_female_items = set(top_male_items).union(set(top_female_items))  # Items in male or female list

    # Items not in male or female list
    non_correlated_items = all_items - male_female_items

    # Ensure `top_male_items` are added first, then non-correlated items
    # 1. Convert `top_male_items` and `non_correlated_items` to lists (to maintain order)
    ordered_male_item_list = top_male_items + list(non_correlated_items)

    # 2. Create the `male_item` set using the ordered list
    male_item = set(ordered_male_item_list)  # Create set from the ordered list

    # Do the same for female items, ensuring `top_female_items` appear first
    ordered_female_item_list = top_female_items + list(non_correlated_items)
    female_item = set(ordered_female_item_list)

    # Find common items
    c = set(ordered_male_item_list) & set(ordered_female_item_list)

    # Print the results
    print(len(c))
    print(f"Male: {len(male_item)} & Female: {len(female_item)}")

    # Save results to files
    np.savetxt(f"ml-{data_version}/Other/common_ml{data_version}_RF_Lm_Item.dat", ordered_male_item_list,
               delimiter=", ", fmt="%s")
    np.savetxt(f"ml-{data_version}/Other/common_ml{data_version}_RF_Lf_Item.dat", ordered_female_item_list,
               delimiter=", ", fmt="%s")



def findDiff():

    data_version = 'yahoo'

    # if data_version == '1m':
    #     dir = 'ml-1m/Other/ml1m'
    # elif data_version == '100k':
    #     dir = 'ml-100k/Other/ml100k'

    dir = 'ml-'+str(data_version)+'/Other/ml'+str(data_version)
    print("---- female ----")
    Lf_LR = np.loadtxt(dir+'_Lf_Item.dat', dtype=int)
    Lf_RF = np.loadtxt(dir+'_RF_Lf_Item.dat', dtype=int)
    common_LRRF = [item for item in Lf_LR if item in Lf_RF]
    # common_RFLR = [item for item in Lf_RF if item in Lf_LR]
    # common = [item for item in L_ff if item in common_LRRF] # -- this line only for ml1m
    # print(f"Lf_LR: {len(Lf_LR)} & Lf_RF: {len(Lf_RF)} & L_ff: {len(L_ff)} & common: {len(common)}, common_LRRF:{len(common_LRRF)}")
    print(common_LRRF)

    print("---- male ----")
    Lm_LR = np.loadtxt(dir+'_Lm_Item.dat', dtype=int)
    Lm_RF = np.loadtxt(dir+'_RF_Lm_Item.dat', dtype=int)
    com_LRRF  = [item for item in Lm_LR if item in Lf_RF]
    #com_RFLR = [item for item in Lm_RF if item in Lf_LR]
    #com = [item for item in L_mm if item in com_LRRF] # -- this line only for ml1m
    #print(f"Lm_LR: {len(Lm_LR)} & Lm_RF: {len(Lm_RF)} & L_mm: {len(L_mm)} & common: {len(com)}, com_LRRF:{len(com_LRRF)}")
    print(com_LRRF)

# Run the function using Random Forest Classifier

#find_Corelated_items_by_gender()
#find_Corelated_items_by_gender_rf()
findDiff()

#find_genre_of_correlatedItems()
