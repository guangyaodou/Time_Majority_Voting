from calendar import c
from fileinput import filename
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

score_dict = {}
names = [  
#         'GradientBoostingRegressor',
        'LDA',
#         'Nearest Neighbors',
        'AdaBoostClassifier',
        'RandomForest',
#         "Linear SVM",
#         "RBF SVM",
        "Decision Tree",
        "sLDA",
#         "MLP",
#         'RUSBoost',
#         'Shrinkage LDA',
        ]

def addScores(start, end):
    '''
    The purpose of this function is to populate the score_dict variable with all scores for each subject and model

    start: The id of the first subject to include his/her data
    end: The id of the last subject to include his/her data
    
    return: None
    Side-effect: populates the score_dict with the accuracy for each subject id and algorithm
    '''

    File_name = "output/" + '/scores_' + str(start) + '_' + str(end) + '_subjects.json'

    File = open(File_name)
    Data = json.load(File)
    File.close()
    
    for subject_id in Data:
        #print(subject_id)
        if subject_id in score_dict: 
            continue
        score_dict[subject_id] = Data[subject_id]
        

def compute_model_Scores(y_names, subject_ids):
    '''
    y_names: a list of models to compute their average
    subject_ids: a list of subject_ids to indicate which 
                 subjects to include when calculating the average

    return a dictionary (map) that maps each algorithm to its average accuracy over the subjects
    '''

    average_accuracy_recorder = {}
    for y_name in y_names:
        y_accuracy = []
        for key in subject_ids:
            if y_name == "TMV":
                y_accuracy.append(score_dict[key][y_name])
            else:
                y_accuracy.append(score_dict[key][y_name][0])
        average_of_y_name = sum(y_accuracy) / float(len(y_accuracy))
        average_accuracy_recorder[y_name] = average_of_y_name
    return average_accuracy_recorder


def computer_subject_Scores(models, subject_ids):

    '''
    subject_ids: a list of subject_ids to computer their average score

    models: a list of models to include in calculating the average for each a subject

    return a dictionary (map) that maps each subject to its average accuracy over the included models
    '''

    subject = {}
    for subject_id in subject_ids:
        sum = 0
        for model in models:
            if model == 'TMV':
                sum += score_dict[subject_id][model]
            else:
                sum += score_dict[subject_id][model][0]

        average = sum / 3
        subject[subject_id] = average
    
    return subject

def plot(subject_ids, y_names, model_scores, fileName):

    '''
    Plot subject_ids on x-axis versus their scores on the y-axis

    subject_ids: which subjects to plot
    y_names: which models to plot a line for (each model has a different line)

    *** The function will just show the plot and won't save it; you have to manually press the save button yourself

    '''

    x_axis = [int(s_id) for s_id in subject_ids]
    plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots()

    for y_name in y_names:
        y_accuracy = []
        for key in subject_ids:
            if y_name == "TMV":
                y_accuracy.append(score_dict[key][y_name])
            else:
                y_accuracy.append(score_dict[key][y_name][0])
        average_of_y_name = model_scores[y_name]
        ax.plot(x_axis, y_accuracy, label = y_name + "("+ str(round(average_of_y_name, 2))+")")


    #ax.set_position([0.1, 0.5, 1, 1])
    ax.legend(loc = 'lower right')
    plt.axhline(y = 0.5, color='r', linestyle=':')

    plt.xlabel('Subject IDs', fontsize = 15)
    plt.ylabel('Accuracy', fontsize = 15)

    start = 1
    end = 370
    ax.set_xticks(np.arange(start, end, 46))
    ax.set_yticks(np.arange(0.1, 1.1, 0.2))
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.savefig(fileName, bbox_inches='tight', dpi = 800)


def removeOutliers(subject_scores, threshold, subject_ids):

    '''
    returns two lists: a list of the ids of the remaining subjects and another one with the ids of the removed subjects
    
    -> subjects are removed if their average score is beyond the threshold (specified when the function is called) 
    '''

    filtered_subjects = []
    removed_subjects = []
    
    for subject_id in subject_ids:
        if subject_scores[subject_id] > threshold:
            filtered_subjects.append(subject_id)
        else:
            removed_subjects.append(subject_id)

    return filtered_subjects, removed_subjects


def makeTable(models, model_scores, fileName):
    '''
    This functions makes a table and saves it in the same folder under the name of fileName
    '''

    avg_accuracy = []
    for model in models:
        avg_accuracy.append(round(model_scores[model], 3))

    data={ "Average_Accuracy": avg_accuracy}
    df = pd.DataFrame(data, index = models)
    df = df.sort_values(by=["Average_Accuracy"], ascending=False)
    df.to_csv(fileName)


#### This is where the function calls and the coding logic starts ####


#The next lines populate score_dict with all data for all subjects from 1 to 369
addScores(1, 369)    


#store the included subject_ids in a list to use in the future
subject_ids = [key for key in score_dict]



###############################################
#computing the scores for all algorithms using all subjects and sorting them in descending order 
y_names = names.copy()
y_names.append("TMV")
model_scores = compute_model_Scores(y_names, subject_ids = subject_ids)
y_names.sort(key = lambda x : -model_scores[x])
###############################################

best_3 = y_names[:3] #Take the best 3 classifiers based on the average accuracy


#make a table for the accuracies of the best 3 models and save it in a csv file
makeTable(models = y_names, model_scores = model_scores, fileName = "Top_3_accuracy.csv")


#make a plot that included data for all subjects and the best 3 models
plot(subject_ids = subject_ids, y_names = best_3, model_scores = model_scores, fileName = "EEGEyeNet_wl_outliers")


#compute the average score for each subject based on the average of the top 3 algorithms
subject_scores = computer_subject_Scores(models = best_3, subject_ids = subject_ids)


#create a sorted list (ascending) of subject ids based on their top 3 average scores
sorted_subject_ids = [key for key in score_dict]
sorted_subject_ids.sort(key = lambda x : subject_scores[x])



#remove subjects whose average score is less than the threshold
filtered_subjects, removed_subjects = removeOutliers(subject_scores=subject_scores, threshold = .6, subject_ids=subject_ids)


#make a table for the removed subjects and their accuracies and save it
makeTable(removed_subjects, subject_scores, "Outliers.csv")


#recomputer model scores after removing subject outliers
model_scores = compute_model_Scores(y_names, filtered_subjects) #only includes filtered_subjects
y_names.sort(key = lambda x : -model_scores[x]) #sort
best_3 = y_names[:3] #take the best 3


#replot after removing outliers
plot(filtered_subjects, best_3, model_scores = model_scores, fileName = "plot_after_removal.png")

#remake the table of accuracies after removing the outliers
makeTable(models = y_names, model_scores = model_scores, fileName = "Top_3_accuracy_outliers_removed.csv")
