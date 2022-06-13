import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# metrics are used to find accuracy or error
from sklearn import metrics

data_folder_name = "Raw_2018"

plateau_threashold = 14


def combine_folds(folds_dict, fold_num):
    folds = []
    for key in folds_dict:
        if key != fold_num:
            folds.append(folds_dict[key])
    res = pd.concat(folds)
    return res


## checks if the file_name exists in the folder_name
def check_file_exists(folder_name, file_name):
    return os.path.exists(folder_name + "/" + file_name)


# compares two rows to see if they are the same in the pandas dataframe
def compare_rows(row_1, row_2):
    row_diff = row_1 == row_2
    for diff in row_diff:
        if diff == False:
            return False
    return True


def mark_zero_plateau(data, start, end):
    idx = start
    while idx < end:
        num_cols = len(data.loc[idx])
        data.loc[idx] = [0] * num_cols
        idx += 1


def findPlateau(data, plateau_threashold: int):
    i = 0
    while i < len(data) - 1:
        row_1 = data.iloc[i, :]
        row_2 = data.iloc[i + 1, :]
        same = compare_rows(row_1, row_2)
        if same:
            counter = 2
            finished = False
            while not finished:
                idx = i + counter
                if idx >= len(data):
                    break
                temp_row = data.iloc[idx, :]
                if not compare_rows(row_2, temp_row):
                    finished = True
                else:
                    counter += 1
            if counter >= plateau_threashold:
                mark_zero_plateau(data, i, i + counter - 1)
            i += counter
        else:
            i += 1


def find_transitions(data, seconds):
    last_task_left = len(data) - 9000
    num_rows = seconds * 10
    for i in range(4):
        starting_pos = 3000 * i
        mark_zero_plateau(data, starting_pos, starting_pos + num_rows)


def is_zero(data, pos):
    num_cols = len(data.loc[pos])
    compare = data.iloc[pos] == [0] * num_cols
    for bol in compare:
        if bol == False:
            return False
    return True


def update_data_after_plateau(data):
    i = 0
    index_drop = []
    while i < len(data):
        if is_zero(data, i):
            index_drop.append(i)
        i += 1
    data = data.drop(labels=index_drop, axis=0)
    data.reset_index(drop=True, inplace=True)
    return data


def session_ground_truth(order, last_task_left):
    res_lst = []
    for i in range(len(order) - 1):
        res_lst.extend([order[i]] * 3000)
    res_lst.extend([order[-1]] * last_task_left)
    return res_lst


# define models to train
names = [
    'GradientBoosting',
    'LDA',
    'Nearest Neighbors',
    'AdaBoostClassifier',
    'RandomForest',
    "Linear SVM",
    'RBF SVM',
    'Decision Tree',
    'Shrinkage LDA',
]

# build classifiers
classifiers = [
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=4),
    AdaBoostClassifier(),
    RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
]

subject_id_lst = [55, 56, 57, 58, 59, 61, 66, 67, 68, 70, 71, 72]
session_lst = [1, 2, 3, 4, 5]

# add ground truth in the end of read_file
# Math (M): 1; Read(R): 3,
# Shut (S): 2; Open(O): 4
# All sessions order: 1, 2, 3, 4

session_task_order = {}
session_task_order[1] = [1, 2, 3, 4]
session_task_order[2] = [1, 2, 3, 4]
session_task_order[3] = [1, 2, 3, 4]
session_task_order[4] = [1, 2, 3, 4]
session_task_order[5] = [1, 2, 3, 4]

subject_id_start = 1
subject_id_end = 12  # GRE_Relax max 12

subject_algorithms_dict = {}

for subject_id in range(subject_id_start, subject_id_end + 1):

    print("Running subject id", subject_id)

    subject_id_num = subject_id_lst[subject_id - 1]

    read_file_session_1 = None
    read_file_session_2 = None
    read_file_session_3 = None
    read_file_session_4 = None
    read_file_session_5 = None

    # for session in session_lst:
    for session in session_lst:
        file_name = str(subject_id_num) + '_' + str(session) + '.txt'
        file_exists = check_file_exists(data_folder_name, file_name)
        read_file = None
        if file_exists:
            read_file = pd.read_csv(data_folder_name + '/' + file_name, delim_whitespace=True, header=None)
            read_file = read_file.iloc[:, 1:]
            if len(read_file) > 12000:
                read_file = read_file.iloc[:12000, :]
        session_order = session_task_order[session]
        if read_file is not None:
            last_task_left = len(read_file) - 9000
            ground_truth = session_ground_truth(session_order, last_task_left)
            read_file['y'] = ground_truth

            findPlateau(read_file, plateau_threashold)

            find_transitions(read_file, 18)

            if session == 1:
                read_file_session_1 = update_data_after_plateau(read_file)
            elif session == 2:
                read_file_session_2 = update_data_after_plateau(read_file)
            elif session == 3:
                read_file_session_3 = update_data_after_plateau(read_file)
            elif session == 4:
                read_file_session_4 = update_data_after_plateau(read_file)
            else:
                read_file_session_5 = update_data_after_plateau(read_file)

    session_data = [read_file_session_1, read_file_session_2, read_file_session_3, read_file_session_4,
                    read_file_session_5]

    i = 0
    length = len(session_data)

    # remoove sessions that are noisy
    while i < length:
        temp_data = session_data[i]
        if temp_data is None:
            session_data.remove(temp_data)
            i -= 1
            length -= 1
            continue
        if temp_data is not None:
            if len(temp_data) < 0.35 * 12000:
                print("We are excluding session", i + 1)
                session_data.remove(temp_data)
                i -= 1
                length -= 1
        i += 1

    if len(session_data) == 0:
        print("We are excluding subject", subject_id)
        continue

    data = pd.concat(session_data)
    data.reset_index(drop=True, inplace=True)

    # cut 6 folds
    test1data = []
    test2data = []
    test3data = []
    test4data = []
    test5data = []
    test6data = []

    folds_dict = {}
    folds_dict[0] = test1data
    folds_dict[1] = test2data
    folds_dict[2] = test3data
    folds_dict[3] = test4data
    folds_dict[4] = test5data
    folds_dict[5] = test6data

    task_data = None

    for task in range(4):
        task_data = data[data['y'] == task + 1]
        total_rows = len(task_data)
        per_folds = total_rows // 6
        for i in range(6):
            starting = per_folds * i
            if i != 5:
                folds_dict[i].append(task_data.iloc[starting: starting + per_folds])
            else:
                folds_dict[i].append(task_data.iloc[starting:])

    for key in folds_dict:
        folds_dict[key] = pd.concat(folds_dict[key])
        folds_dict[key].reset_index(drop=True, inplace=True)

    accuracy_dict = {}

    models = zip(names, classifiers)
    for name, classifier in models:
        accuracy = 0
        for fold_num in range(len(folds_dict)):
            data_train = combine_folds(folds_dict, fold_num)
            X = data_train.iloc[:, :-1]
            y = data_train.iloc[:, -1]
            clf = classifier
            clf.fit(X, y)
            data_test = folds_dict[fold_num]
            X_test = data_test.iloc[:, :-1]
            y_test = data_test.iloc[:, -1]
            y_predict = []
            if name == "GradientBoostingRegressor":
                accuracy += clf.score(X_test, y_test)
            else:
                y_predict = clf.predict(X_test)
                accuracy += metrics.accuracy_score(y_test, y_predict)
        accuracy_dict[name] = accuracy / len(folds_dict)

    subject_algorithms_dict[subject_id] = accuracy_dict

print(subject_algorithms_dict)

algorithm_sum_dict = {}
for name in names:
    if name not in algorithm_sum_dict:
        algorithm_sum_dict[name] = 0
    for i in range(subject_id_start, subject_id_end + 1):
        algorithm_sum_dict[name] += subject_algorithms_dict[i][name]

classifier_name = names.copy()
classifier_name.sort(key=lambda x: algorithm_sum_dict[x], reverse=True)
best_classifier_name = classifier_name[0]
print("order of the classifier is", classifier_name)

subject_id_order = [i for i in range(subject_id_start, subject_id_end + 1)]
subject_id_order.sort(key=lambda x: subject_algorithms_dict[x][best_classifier_name])
print("subject id order is", subject_id_order)

for key in algorithm_sum_dict:
    algorithm_sum_dict[key] /= len(subject_id_order)

# x_axis = np.arange(len(subject_id_order))
x_axis = list(range(len(subject_id_order)))

plt.figure(figsize=(10, 5))
fig, ax = plt.subplots()
for classifier in classifier_name:
    y = []
    for s_id in subject_id_order:
        y.append(subject_algorithms_dict[s_id][classifier])
    label_name = classifier
    if classifier == "Shrinkage LDA":
        label_name = "sLDA"
    if classifier == "Nearest Neighbors":
        label_name = "KNN"
    if classifier == "AdaBoostClassifier":
        label_name = "AdaBoost"
    ax.plot(x_axis, y, marker='D', label=label_name + "(" + str(round(algorithm_sum_dict[classifier], 2)) + ")")

ax.set_position([0.1, 0.5, 1.2, 1.0])
ax.legend(loc='upper left')
plt.axhline(y=0.25, color='r', linestyle=':')

plt.xticks(x_axis, subject_id_order, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Subject ID orderd by ' + best_classifier_name, fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.savefig("output/" + data_folder_name + "/algorithm_comparison_each_subject.jpg", bbox_inches='tight', dpi=1500)
plt.show()
