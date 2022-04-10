# Time_Majority_Voting
Time Majority Voting is a new algorithm that can better predict EEG datasets. 

## Datasets

The first step is to download the dataset from this [google drive](https://drive.google.com/drive/u/1/folders/1dM5Lk2oBpfJrz6ByfYemG9eNkKJxpsAj).

You should download all the folders in the google drive provided and copy these folders in the [data_preprocess](data_preprocess). After this step, your [data_preprocess](data_preprocess) should have these folders: old_rwt_plateau_removed_data, old_tcr_plateau_removed_data, rwt_raw_data, and tcr_raw_data.

### What is preprocess.m for?
The file [preprocess.m](data_preprocess/preprocess.m) was used to convert raw EEG datasets to human-readable data such as the csv files you see in the old_rwt_plateau_removed_data and old_tcr_plateau_removed_data. Right now, you can ignore this and just download the datasets directly from the google drive link provided.

## Analyze State-of-the-art algorithms

Next step is to run [analyze_algorithms.ipynb](analyze_algorithms.ipynb) to execute the state-of-the-art machine learning algorithms. All of the results will be stored in [tcr_old_results](tcr_old_results). Please verify that the top two algorithms are Random Forest and RBF SVM. 

Side node: It took me around 40 - 50 minutes to finish running [analyze_algorithms.ipynb](analyze_algorithms.ipynb). Running time varies depending on different computers.

## Time Majority Voting Algorithm

Last, run [Time_majority_voting_algorithm.ipynb](Time_majority_voting_algorithm.ipynb) to execute the TMV algorithm, and all of the results/graphs will be stored in [Time_majority_results](Time_majority_results). Please verify the increase of accuracy comparing to the Random Forest, and you procuced similar graphs in the paper. The link to the paper can be found on the [Research Tasks Forum](https://xiaodongqu.com/viewtopic.php?f=9&t=182&p=757#p757).

Side node: It took me around 10 - 20 minutes to finish running [Time_majority_voting_algorithm.ipynb](Time_majority_voting_algorithm.ipynb). Running time varies depending on different computers.
