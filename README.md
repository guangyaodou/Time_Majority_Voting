# Time_Majority_Voting
Time Majority Voting is a new algorithm that can better predict EEG datasets. 

## Datasets

The first step is to download the dataset from this [google drive](https://drive.google.com/drive/u/1/folders/1dM5Lk2oBpfJrz6ByfYemG9eNkKJxpsAj).

You should download all the folders in the google drive provided and copy these folders in the [data_preprocess](data_preprocess). After this step, your [data_preprocess](data_preprocess) should have these folders: old_rwt_plateau_removed_data, old_tcr_plateau_removed_data, rwt_raw_data, and tcr_raw_data.

Also, please download the csv file version of the EEGEyeNet dataset (Abdel has the dataset).

### What is preprocess.m for?
The file [preprocess.m](data_preprocess/preprocess.m) was used to convert raw EEG datasets to human-readable data such as the csv files you see in the old_rwt_plateau_removed_data and old_tcr_plateau_removed_data. Right now, you can ignore this and just download the datasets directly from the google drive link provided.

## Analyze State-of-the-art algorithms

You can run [analyze_algorithms.ipynb](analyze_algorithms.ipynb) to execute the state-of-the-art machine learning algorithms. All of the results will be stored in [tcr_old_results](tcr_old_results). Please verify that the top two algorithms are Random Forest and RBF SVM. 

Side node: It took me around 40 - 50 minutes to finish running [analyze_algorithms.ipynb](analyze_algorithms.ipynb). Running time varies depending on different computers.

## Time Majority Voting Algorithm for EEGEyeNet

Run [TMV_EEGEyeNet.ipynb](TMV_EEGEyeNet.ipynb) to execute the TMV algorithm on the EEGEyeNet dataset. You can also decide to run which subjects by adjusting the variable "total_subjects". You can uncomment and comment out the corresponding names and classifiers in the second block to determine what classifiers you want to run. 

Side Note: There are 369 subjects in total. Run time varies based on number of subjects you are going to run.

## Time Majority Voting Algorithm for TCR and RWT dataset

Run [Time_majority_algorithm.ipynb](Time_majority_algorithm.ipynb) to execute the TMV algorithm on the TCR and the RWT dataset, and all of the results/graphs will be stored in [Time_majority_results](Time_majority_results). 

Inside the [Time_majority_results](Time_majority_results) folder, remeber to create a folder named "rwt" and a folder named "tcr". Inside each of these folders, create a folder called "TMV". Change the data_source to be either 'rwt' or 'tcr'. You can also decide to run which subjects by adjusting the variables "subject_id_start" and "subject_id_end". You can uncomment and comment out the corresponding names and classifiers in the second block to determine what classifiers you want to run. 

Please verify the increase of accuracy comparing to the Random Forest, and you procuced similar graphs in the paper. The link to the paper can be found on the [Research Tasks Forum](https://xiaodongqu.com/viewtopic.php?f=9&t=182&p=757#p757).

Side node: It took me around 10 - 20 minutes to finish running [Time_majority_algorithm.ipynb](Time_majority_algorithm.ipynb). Running time varies depending on different computers.
