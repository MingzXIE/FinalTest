import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Store the algorithms into a dictionary
clf_all= {'tree':DecisionTreeClassifier(),
          'svm':SVC(),
          'RDtree':RandomForestClassifier(n_jobs = -1),
          'logit':LogisticRegression(),
          'bayes':GaussianNB()
          }


# Read Data
def read_file(filename, n_fold_input = 5):
    data = pd.read_csv(filename, sep='\t', header=None)
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    n_fold = n_fold_input  # n_fold
    # Split data by StratifiedKFold
    kf = StratifiedKFold(data_y, n_fold, random_state=2018)

    return data_x, data_y, kf


def classifier_test(clf, kf, n_fold, data_x, data_y):

    # clf represents the types of the classifier
    # adopt k-fold cross validation to test get the accuracy of each algorithms

    list_result = []
    for train_index,test_index in kf:
        train_x,train_y = data_x.iloc[train_index],data_y.iloc[train_index] # training data
        test_x,test_y = data_x.iloc[test_index],data_y.iloc[test_index] # testing data

        # fit the data
        clf.fit(train_x,train_y)
        # make prediction
        predict_y = clf.predict(test_x)
        # test
        list_result.append(accuracy_score(test_y,predict_y))
    # return the average accuracy of each algorithm
    return sum(list_result)/n_fold


# select the best algorithm/classifier for the user
def select_best_algorithm(filename, n_fold_input = 5):
    clf_all_label = []  # The name of the classifier
    clf_all_result = []  # The result of prediction
    accuracy_list = ''
    matrix, labels, kf = read_file(filename, n_fold_input)
    for k, v in clf_all.items():
        # Training and Testing
        clf_all_label.append(k)
        accuracy_tmp = classifier_test(v, kf, n_fold_input, matrix, labels)
        clf_all_result.append(accuracy_tmp)
        accuracy_list += str(k) + ": " + str(accuracy_tmp) + "\n"
    # select best algorithm via the accuracy
    best_choice_index = clf_all_result.index(max(clf_all_result))
    best_algorithm = clf_all_label[best_choice_index]
    highest_accuracy = max(clf_all_result)
    return accuracy_list, highest_accuracy, best_algorithm


def input_and_predict(best_algorithm, unlabeled_data):
    clf_best = clf_all[best_algorithm]
    data_to_predict = np.array(string_to_list(unlabeled_data)).reshape(1, -1)
    result = clf_best.predict(data_to_predict)
    return result[0]


def string_to_list(string_input):
    line = string_input.strip()
    float_list=[]
    elements_list = line.split(',')
    for str_number in elements_list:
        float_num = float(str_number)
        float_list.append(float_num)
    return float_list

a,b,c = select_best_algorithm("datingTestSet2.txt")
print(a)
print(b)
print(c)

new_data = "44952,3.424649,1.004504"
print(input_and_predict(c, new_data))