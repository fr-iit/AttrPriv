import pickle
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def shuffle_two_arrays(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a_new = np.zeros(shape=a.shape)
    b_new = np.zeros(shape=b.shape)
    masking = np.arange(len(a))
    np.random.shuffle(masking)
    for index in range(len(a)):
        a_new[index, :] = a[masking[index], :]
        b_new[index] = b[masking[index]]
    return a, b


def confusion_matrix(y, t, size=2, image=False):
    print("Confusion Matrix")
    # print(y,t)

    matrix = np.zeros(shape=(size, size))
    for ys, ts in zip(y, t):
        matrix[int(ys), int(ts)] += 1
    print(matrix)
    if image:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')


def classification_accuracy(Y, T):
    correct = 0
    for y, t in zip(Y, T):
        if y == t:
            correct += 1
    return correct / len(T)


def one_hot(x, length):
    out = np.zeros(shape=(length,))
    out[x] = 1
    return np.asarray(out)


def plot_line(x, y, show=True, legend=[], xlabel="", ylabel="", ):
    plt.plot(x, y)
    if show:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()


def ROC_cv_obf(X, X_obf, T, classifier, show_plot=True):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    #from scipy import interp
    from sklearn.metrics import balanced_accuracy_score

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=11)

    tprs = []
    aucs = []
    p_r_aucs = []
    mean_fpr = np.round(np.linspace(0, 1, 100),2)
    recalls = []
    precisions = []
    accuracies = []
    baccuracies = []
    i = 0
    for train, test in cv.split(X, T):
        print(i)
        x, t = X[train], T[train]
        classifier.fit(x, t)
        Y = classifier.predict(X_obf[test])
        TPR, TNR, FPR, FNR, precision, accuracy = performance_measures(Y, T[test])
        bacc = balanced_accuracy_score(Y, T[test])
        recalls.append(TPR)
        precisions.append(precision)
        accuracies.append(accuracy)
        baccuracies.append(bacc)
        probas_ = classifier.predict_proba(X_obf[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(T[test], probas_[:, 1])
        # print("thresholds", thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        #plt.subplot(1,2,1)
        if show_plot:
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    print("CV recall:", np.average(recalls), "+-", np.std(recalls), "CV precision:", np.average(precisions), "+-",
          np.std(precisions))
    print("CV accuracy:", np.average(accuracies), np.std(accuracies))
    print("CV baccuracy:", np.average(baccuracies), np.std(baccuracies))

    #plt.subplot(1,2,1)
    if show_plot:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.round(np.mean(tprs, axis=0), 2)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(f"mean_fpr= {list(mean_fpr)}")
    print(f"mean_tpr= {list(mean_tpr)}")
    print(f"mean_auc= {mean_auc: .2f}")
    print(f"std_auc= {std_auc: .2f}")
    if show_plot:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.show()

    return mean_auc, std_auc

def performance_measures(Y, T):
    tp, fp, fn, tn = 0, 0, 0, 0

    for y, t in zip(Y, T):
        if y == 0 and t == 0:
            tp += 1
        elif y == 1 and t == 0:
            fn += 1
        elif y == 0 and t == 1:
            fp += 1
        elif y == 1 and t == 1:
            tn += 1
    if tp == 0:
        TPR = 0
    else:
        TPR = tp / (tp + fn)  # also called sensitivity/Recall
    if fn == 0:
        FNR = 0
    else:
        FNR = fn / (tp + fn)  # also called miss rate
    if tn == 0:
        TNR = 0
    else:
        TNR = tn / (tn + fp)  # specificity/selectivity
    if fp == 0:
        FPR = 0
    else:
        FPR = fp / (fp + tn)

    print("tp:", tp, ", fp:", fp, "\nfn:", fn, ", tn:", tn)
    if tp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + tn) == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return TPR, TNR, FPR, FNR, precision, accuracy


def ROC_plot(X, T, model):

    import sklearn.metrics as metrics
    probs = model.predict_proba(X)
    preds = probs[:, 1]
    y_pred = model.predict(X)
    from sklearn.metrics import accuracy_score
    print("accuracy:", accuracy_score(T, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(T, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def ROC_cv(X, T, classifier, show_plot=True):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    #from scipy import interp
    from sklearn.metrics import balanced_accuracy_score

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    p_r_aucs = []
    mean_fpr = np.round(np.linspace(0, 1, 100),2)
    recalls = []
    precisions = []
    accuracies = []
    baccuracies = []
    i = 0
    for train, test in cv.split(X, T):
        print(i)
        x, t = X[train], T[train]

        classifier.fit(x, t)
        Y = classifier.predict(X[test])
        TPR, TNR, FPR, FNR, precision, accuracy = performance_measures(Y, T[test])
        bacc = balanced_accuracy_score(Y, T[test])
        recalls.append(TPR)
        precisions.append(precision)
        accuracies.append(accuracy)
        probas_ = classifier.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(T[test], probas_[:, 1])
        # print("thresholds", thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        baccuracies.append(bacc)

        # plt.subplot(1,2,1)
        if show_plot:
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    # plt.subplot(1,2,1)
    if show_plot:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

    mean_tpr = np.round(np.mean(tprs, axis=0), 2)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(f"mean_fpr= {list(mean_fpr)}")
    print(f"mean_tpr= {list(mean_tpr)}")
    print(f"mean_auc= {mean_auc: .2f}")
    print(f"std_auc= {std_auc: .2f}")
    if show_plot:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.show()

    print("CV recall:", np.average(recalls), "+-", np.std(recalls), "CV precision:", np.average(precisions), "+-",
          np.std(precisions))
    print("CV accuracy:", np.average(accuracies))
    print("CV baccuracy:", np.average(baccuracies))
    return mean_auc, std_auc


def ROC_multiclass(X, T, classifier, n_classes=21):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    #from scipy import interp
    print("n_classes", n_classes)
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5)

    tprs = []
    fprs = []
    aucs = []
    p_r_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    recalls = []
    precisions = []
    i = 0
    for train, test in cv.split(X, T):
        classifier.fit(X[train], T[train])
        Y = classifier.predict(X[test])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_i in range(n_classes):
            Y_test = np.zeros(shape=Y.shape)
            for i in range(len(Y)):
                if Y[i] == class_i:
                    Y_test[i] = 1
            T_test = np.zeros(shape=T[test].shape)
            for i in range(len(T[test])):
                if T[test][i] == class_i:
                    T_test[i] = 1

            TPR, TNR, FPR, FNR, precision, accuracy = performance_measures(Y_test, T_test)
            recalls.append(TPR)
            precisions.append(precision)
            probas_ = classifier.predict_proba(X[test])
            # Compute ROC curve and area the curve

            fpr[class_i], tpr[class_i], _ = roc_curve(T_test, probas_[:, class_i])
            roc_auc[class_i] = auc(fpr[class_i], tpr[class_i])
            fpr["micro"], tpr["micro"], _ = roc_curve(T_test.ravel(), Y.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        i += 1

    # Compute macro-average ROC curve and ROC area
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, _color_ in zip(range(n_classes), colors):
        color = plt.cm.get_cmap('hsv', i)
        plt.plot(fpr[i], tpr[i], color=np.random.rand(3, ), lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    print("CV recall:", np.average(recalls), "CV precision:", np.average(precisions))

def chi2_selection(X, T):
    from sklearn.feature_selection import chi2 as CHI2
    chi, pval = CHI2(X, T)
    relevant_features = []
    print(X.shape)
    for index, p in enumerate(pval):
        if p <= 0.001:  # the two variables (T and the feature row) are dependent
            relevant_features.append(X[:, index])
    return np.transpose(np.asarray(relevant_features))


def random_forest_selection(X, T, threshold=0.0001):
    from sklearn.ensemble import ExtraTreesClassifier
    importance = np.zeros(shape=(X.shape[1],))
    for i in range(10):
        model = ExtraTreesClassifier(max_depth=10)
        model.fit(X, T)
        importance += model.feature_importances_
    importance /= 10

    selected = []
    not_selected = []
    counter = 0
    for movie, score in enumerate(importance):
        if score > threshold:
            selected.append(X[:, movie])
            print(movie + 1, end=",")
            counter += 1
    print()
    for movie, score in enumerate(importance):
        if score <= threshold:
            not_selected.append(X[:, movie])

    # print(counter)
    return np.transpose(np.asarray(selected)), np.transpose(np.asarray(not_selected))


def feature_selection(X, T, selection_method):

    _, pval = selection_method(X, T)
    relevant_features = []
    print(X.shape)
    for index, p in enumerate(pval):
        if p >= 0.05 / len(pval):  # the two variables (T and the feature row) are dependent
            # print(index+1, end=",")
            relevant_features.append(X[:, index])
    return np.transpose(np.asarray(relevant_features))


def select_male_female_different(X, T):
    from scipy.stats import ttest_ind
    X = np.transpose(X)
    fs = []
    p_vals = []
    for movie in X:
        male_ratings = []
        female_ratings = []
        for user_index, rating in enumerate(movie):
            if rating > 0:
                if T[user_index] == 1:
                    male_ratings.append(rating)
                else:
                    female_ratings.append(rating)

        f, p_value = ttest_ind(male_ratings, female_ratings)
        p_vals.append(p_value)
        fs.append(f)
    return fs, p_vals


def normalize(X):
    from sklearn import preprocessing
    X = preprocessing.normalize(X)
    return X


def standardize(X):
    from sklearn import preprocessing
    X = preprocessing.scale(X)
    return X


def center(X, axis=0, include_zero=True):
    if axis == 1:
        X = np.transpose(X)

    if include_zero:
        mean = np.mean(X, axis=0)
        X -= mean
    else:
        X = np.transpose(X)
        centered_X = np.zeros(shape=X.shape)
        for index, row in enumerate(X):
            clean_row = []
            for rating in row:
                if rating > 0:
                    clean_row.append(rating)
            if len(clean_row) == 0:
                mean = 0
            else:
                mean = np.mean(clean_row)
            centered_X[index, :] = row - mean
        X = centered_X
        X = np.transpose(X)

    if axis == 1:
        X = np.transpose(X)
    return X


def normalize2(X):
    X = np.transpose(X)
    copy = np.zeros(shape=X.shape)
    for index, row in enumerate(X):
        clean_ratings = []
        for rating in row:
            if rating > 0:
                clean_ratings.append(rating)
        mean = np.mean(clean_ratings)
        std = np.std(clean_ratings)
        if std == 0:
            std = 1
            # print("hello")

        for index_rating, rating in enumerate(row):
            if rating > 0:
                copy[index, index_rating] = (rating - mean) / std
            # print((rating-mean)/std)

    return np.transpose(copy)

def remove_significant_features(X, T):
    items = X.shape[1]
    male_index = np.argwhere(T == 0).reshape(1, -1)[0]
    female_index = np.argwhere(T == 1).reshape(1, -1)[0]

    print(male_index)
    for item in range(items):
        male_ratings = X[np.argwhere(X[male_index, item] > 0).reshape(1, -1)[0]]
        female_ratings = X[female_index, item]
        print(male_index.shape)


def balance_data(X, T):
    males = X[np.argwhere(T == 0)[:, 0]]
    females = X[np.argwhere(T == 1)[:, 0]]
    min_size = min(len(males), len(females))
    new_X = []
    new_T = []
    np.random.seed(0)
    np.random.shuffle(males)
    np.random.shuffle(females)
    males = males[0:min_size]
    females = females[0:min_size]
    for i in range(min_size):
        new_X.append(males[i, :])
        new_T.append(0)
        new_X.append(females[i, :])
        new_T.append(1)

    X = np.asarray(new_X)
    T = np.asarray(new_T)
    return X, T
