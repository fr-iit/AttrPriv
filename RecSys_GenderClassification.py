import Classifiers
import RecSys_Utils as Utils
import numpy as np
import RecSys_DataLoader as DL


# ---- Classifers ----
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def one_million(classifier, data_version = '100k'):

    # Read the needed inputs
    if data_version == '1m':
        print(data_version)
        T = DL.load_gender_vector_1m()
        X = DL.load_user_item_matrix_1m()

    elif data_version == '100k':
        print(data_version)
        T = DL.load_gender_vector_100k()
        X = DL.load_user_item_matrix_100k()

    elif data_version == 'yahoo':
        print(data_version)
        T = DL.load_gender_vector_yahoo()
        X = DL.load_user_item_matrix_yahoo()

    X = Utils.normalize(X)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    print("before", X_train.shape)
    print(X_train.shape)

    classifier(X_train, T_train)

    random_state = np.random.RandomState(0)
    # model = LogisticRegression(penalty='l2', C=1.0, random_state=random_state)  # , class_weight='balanced' penalty='l2', C=545.5594781168514, random_state=random_state) #
    model = SVC(kernel='linear', probability=True, random_state=random_state)
    # model = xgb.XGBClassifier(objective='multi:softmax', num_class=2, random_state=42)
    # model = AdaBoostClassifier(n_estimators=50, random_state=42)
    # model = RandomForestClassifier(n_estimators=100, random_state=0)

    model.fit(X_train, T_train)

    Utils.ROC_plot(X_test, T_test, model)  # ROC_plot


def one_million_obfuscated(classifier, data_version = '100k'):
    print(classifier)
    # Read the needed inputs
    if data_version == '1m':
        print(data_version)
        T = DL.load_gender_vector_1m()
        X1 = DL.load_user_item_matrix_1m()
        X2 = DL.load_user_item_matrix_1m_masked(file_index=0)
    elif data_version == '100k':
        print(data_version)
        T = DL.load_gender_vector_100k()
        X1 = DL.load_user_item_matrix_100k()
        X2 = DL.load_user_item_matrix_100k_masked()
    elif data_version == 'yahoo':
        print(data_version)
        T = DL.load_gender_vector_yahoo()
        X1 = DL.load_user_item_matrix_yahoo()
        X2 = DL.load_user_item_matrix_yahoo_masked()

    print(X1.shape, X2.shape, T.shape)
    # Normalization
    X1 = Utils.normalize(X1)
    X2 = Utils.normalize(X2)

    print(list(X1[0, :]))
    print(list(X2[0, :]))

    # Classification
    #from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.naive_bayes import MultinomialNB


    random_state = np.random.RandomState(0)
    # model = LogisticRegression(penalty='l2', random_state=random_state)  # C=545.5594781168514,
    model = SVC(kernel='linear', probability=True, random_state=random_state)
    # model = RandomForestClassifier(n_estimators=100, random_state=0)
    # model = RandomForestClassifier()
    # model = GaussianNB()
    # model = MultinomialNB()
    Utils.ROC_cv_obf(X1, X2, T, model)

if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()

    # one_million(Classifiers.log_reg)
    # one_million(Classifiers.svm_classifier)
    # one_million(Classifiers.xgb_classifier)
    # one_million(Classifiers.ada_classifier)
    # one_million(Classifiers.rf_classifier)
    # one_million_obfuscated(Classifiers.log_reg)
    one_million_obfuscated(Classifiers.svm_classifier)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


