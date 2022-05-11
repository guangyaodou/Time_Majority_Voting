from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
from sklearn.model_selection import train_test_split
import random

from typing import List
from statistics import mode

data_source = "BCI-Data"
# Gradient Boost, Adabooost, Linear SVM, Decision Tree, sLDA, MLP took very long time to run
# be cautious before you uncomment these algorithms
names = [
#         'GradientBoostingRegressor',
        'LDA',
        'Nearest Neighbors',
#         'AdaBoostClassifier',
        'RandomForest',
        "Linear SVM",
#         "RBF SVM",
        "Decision Tree",
#         "sLDA",
        # "MLP",
        ]

# build classifiers
classifiers = [
#             GradientBoostingRegressor(random_state=1),
            LinearDiscriminantAnalysis(),
            KNeighborsClassifier(n_neighbors=4),
#             AdaBoostClassifier(n_estimators=400, learning_rate = 0.6),
            RandomForestClassifier(n_estimators=300, max_features = "sqrt", oob_score = True),
            SVC(kernel="linear", C=0.025),
#             SVC(gamma=2, C=1),
            DecisionTreeClassifier(),
#             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
#             MLPClassifier(random_state=1, max_iter=300),
              ]

def most_common(List):
    return(mode(List))

# prediction_first is the prediction of the classifier that has higher accuracy
def time_majority_voting(prediction_first : List, prediction_second : List) -> List:
    res = []
    majority = most_common(prediction_first)
    print("Majority y is", majority)
    if len(prediction_first) != len(prediction_second):
        raise Exception("length do not match")
    for i in range(len(prediction_first)):
        node_one = prediction_first[i]
        node_two = prediction_second[i]
        if node_one == node_two:
            res.append(node_one)
        else:
            res.append(majority)
    return res

def calculate_accuracy(y_one : List, y_two : List) -> float:
    if len(y_one) != len(y_two):
        raise Exception("length do not match")
    numerator = 0
    for i in range(len(y_one)):
        first = y_one[i]
        second = y_two[i]
        if first == second:
            numerator += 1
    return float(numerator) / float(len(y_two))


scoring = "accuracy"
total_subjects = 3

score_dict = {}
time_record = {}

tmv_classifier_record = {}

print("start running classifiers")
print("="*20)
for subject_id in range(1, total_subjects + 1):
    df = pd.read_csv(data_source + "/" + str(subject_id) + "-train.csv", header=None)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    col_names = ["y"]
    col_names.extend(cols[1:])
    df.columns = col_names

    # df = df.iloc[:, :1000]
    # numpy object
    data = df.to_numpy()

    # split the dataset
    X = data[:, 1:]
    y = data[:, 0]

    print("=" * 20)
    if (len(X)) == 0:
        print(subject_id, "has not data, and will be excluded for further analysis")
        continue

    print("subject id", subject_id)
    print("length of y is", len(y))
    models = zip(names, classifiers)
    Adaboost_predict = []
    Random_Forest_predict = []
    for name, model in models:
        print("The model running is: " + name)
        time_start = time()
        kfold = RepeatedStratifiedKFold(n_splits=7, n_repeats=7)
        scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        time_end = time()
        if subject_id not in score_dict:
            score_dict[subject_id] = {}
            time_record[subject_id] = {}
            tmv_classifier_record[subject_id] = []
        score_dict[subject_id][name] = (scores.mean(), scores.std())
        time_elapsed = time_end - time_start
        time_record[subject_id][name] = time_elapsed

        tp_data = {"Average_Accuracy": scores.mean(), "Avg runtime(s)": time_elapsed}
        df_tp = pd.DataFrame(tp_data, index=[name])
        df_tp.to_csv("output/" + data_source + "/subject "+str(subject_id)+"_"+name+"_accuracy_time.csv")

        print("The average score of " + name + " is", scores.mean(), "with std of", scores.std())
        print("The time spent to run " + name + " is", time_elapsed)

    classifier_name_order = names.copy()
    classifier_name_order.sort(key=lambda x: score_dict[subject_id][x][0], reverse=True)

    classifiers_TMV = classifier_name_order[:2]

    tmv_classifier_record[subject_id].extend(classifiers_TMV)
    print("classifiers that will perform TMV are", classifiers_TMV)

    name_one = classifiers_TMV[0]
    index_one = names.index(name_one)
    name_two = classifiers_TMV[1]
    index_two = names.index(name_two)

    print()
    print("Start performing TMV")
    model_one = classifiers[index_one]
    model_two = classifiers[index_two]

    time_tmv_start = time()
    y_predict_first = cross_val_predict(model_one, X, y, cv=7)
    y_predict_second = cross_val_predict(model_two, X, y, cv=7)

    tmv_predict = time_majority_voting(y_predict_first, y_predict_second)

    tmv_accuracy = calculate_accuracy(tmv_predict, y)
    time_tmv_end = time()

    score_dict[subject_id]["TMV"] = (tmv_accuracy)
    time_record[subject_id]["TMV"] = time_tmv_end - time_tmv_start

    # Uncomment the following line if you want to see results for each subjects
    # tp_data = {"Average_Accuracy": tmv_accuracy, "Avg runtime(s)": time_tmv_end - time_tmv_start}
    # df_tp = pd.DataFrame(tp_data, index=["TMV"])
    # df_tp.to_csv("output/" + data_source + "/subject " + str(subject_id) + "_TMV_accuracy_time.csv")

    print("The time spent to run TMV is", str(time_tmv_end - time_tmv_start))

print("score_dict is", score_dict)
print()
subject_ids = [key for key in score_dict]
print("subject ids are", subject_ids)
print()
y_names = names.copy()
y_names.append("TMV")
# y_names.append("Voting Classifier")
print("classifier names", y_names)

subject_ids.sort(key = lambda x : score_dict[x]["RandomForest"][0])
print(subject_ids)

average_accuracy_recorder = {}
print("y_names", y_names)
for y_name in y_names:
    y_accuracy = []
    for key in subject_ids:
        if y_name == "TMV":
            y_accuracy.append(score_dict[key][y_name])
        else:
            y_accuracy.append(score_dict[key][y_name][0])
    average_of_y_name = sum(y_accuracy) / float(len(y_accuracy))
    average_accuracy_recorder[y_name] = average_of_y_name

y_names.sort(key = lambda x : -average_accuracy_recorder[x])
print("sorted y_names", y_names)

x_axis = [str(s_id) for s_id in subject_ids]
plt.figure(figsize=(10, 5))
fig, ax = plt.subplots()

for y_name in y_names:
    y_accuracy = []
    for key in subject_ids:
        if y_name == "TMV":
            y_accuracy.append(score_dict[key][y_name])
        else:
            y_accuracy.append(score_dict[key][y_name][0])
    average_of_y_name = average_accuracy_recorder[y_name]
    ax.plot(x_axis, y_accuracy, marker='D', label = y_name + "("+str(round(average_of_y_name, 2))+")")

ax.set_position([0.1,0.5, 1.2, 1.0])
ax.legend(loc='lower right')
plt.axhline(y=0.5, color='r', linestyle=':')
plt.xlabel('Subject IDs sorted by the Random Forest', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("output/"+data_source+"/compare_all_subjects.jpg", bbox_inches='tight', dpi = 800)
plt.show()

classifer_name_list = []
avg_accuracy = []
avg_runtime = []
print(time_record)
for key in average_accuracy_recorder:
    classifer_name_list.append(key)
    avg_accuracy.append(round(average_accuracy_recorder[key],3))
    time = 0
    for subject in time_record:
        time += time_record[subject][key]
    time = time / len(subject_ids)
    avg_runtime.append(round(time,3))

data={"Average_Accuracy":avg_accuracy, "Avg runtime(s)": avg_runtime}
df = pd.DataFrame(data, index = classifer_name_list)
df = df.sort_values(by=["Average_Accuracy"], ascending=False)
df.to_csv("output/" +data_source+"/classifier_accuracy_runtime.csv")
print(df)

print(tmv_classifier_record)
print("="*20)
tmv_algorithm_first = []
tmv_algorithm_second = []
ids = []

for key in tmv_classifier_record:
    ids.append(key)
    algorithms = tmv_classifier_record[key]
    tmv_algorithm_first.append(algorithms[0])
    tmv_algorithm_second.append(algorithms[1])

data_algorithms = {"First Algorithm Used":tmv_algorithm_first, "Second Algorithm Used":tmv_algorithm_second}
df_2 = pd.DataFrame(data_algorithms, index = ids)
df_2.to_csv("output/"+data_source+"/classifiers_used_TMV.csv")
print(df_2)