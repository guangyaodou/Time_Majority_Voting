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

**Note**: the first four bullet points will guide you to download the dataset and reformat it as a CSV file with the format we planned our code on. If you face difficulties with these steps or want to skip them, contact the authors, and they will send you the CSV file directly to start from the fifth bullet point.

1 - To get the data into the expected format for our code to run properly, first, you should go to this link https://github.com/ardkastrati/EEGEyeNet, download this GitHub repository on your device, and add the file (process_eegeye.py), which you can find in our code folder, to the folder you just downloaded from the GitHub. Add an empty folder named **data** to the same folder. 

2 - Next, go to this link: https://osf.io/ktv7m/. There, on the left, you will find two folders, the first of which is named "Dropbox: EEGEyeNet." If this first folder has a (+) sign next to it, click on the (+) sign. Otherwise, go to the next step directly. 

3 - Now, you should see four folders pop up. Open the folder named **Prepared** (Click the + sign next to the folder named "prepared"). When you do that, you will see many files showing up. Scroll down till you see four files that start with "LR_task_with", download the last one of them on your device, and then move it to the **data** folder you just created above.

4 - Run the (process_eegeye.py). After the code finishes running, you should see a new CSV file created in the same directory named **EEGEyeNet-data.csv**.

5 - Now, put the CSV file, which you just generated, in our coding directory, where the [README.md] and all other TMV files exist. Then, Run [TMV_EEGEyeNet.ipynb](TMV_EEGEyeNet.ipynb) to execute the TMV algorithm on the EEGEyeNet dataset. You can also decide to run which subjects by adjusting the variables "start_subject" and "end_subject." You can uncomment and comment out the corresponding names and classifiers in the second block to determine what classifiers you want to run. 

6 - Once the code finishes running, you should go to the (EEGEyeNet_analysis) folder and check the (output) folder inside. In the (output) folder, you will see a JSON file that includes all accuracies and standard deviations for all algorithms applied to all subjects. (You don't need to do anything with the file, take a look if you are curious)

7 - Go back to the (EEGEyeNet_analysis) folder, and run [EEGEyeNet_analysis.py] to generate the two plots and the 3 CSV files that document the results in our paper. All plots and files will be generated in the same directory. So, once generated, you should access them easily.


Side Note: There are 369 subjects in total. Run time varies based on the number of subjects you are going to run.

## Time Majority Voting Algorithm for BCI Competition
The first step is to download the dataset from this [google drive](https://drive.google.com/drive/u/0/folders/1H-JAAqDg-2NwOyvTm4l1OuRgp6X0zOpB); the folder is called BCI-Data. You should put this folder in the same directory as your TMV_BCI_Competition.py file. Then, you should create a new folder called "output" and in this folder create another empty folder named "BCI-Data". 

Run [TMV_BCI_Competition.py](TMV_BCI_Competition.py) to execute the TMV algorithm on the BCI competition dataset. You can decide to run which subjects by adjusting the variable "total_subjects" and the starting position of the for loop on line 94. Currently, it runs subjects 1 to 5. You can uncomment and comment out the corresponding names and classifiers in the second block to determine what classifiers you want to run. 

If you want to change the name of your dataset folder, you should change the variable "data_source" (line 23) to the new name. Also, you need to change the name of the folder you created inside the output folder, where the results will be stored, to the new name.

The [output/BCI-Data](output/BCI-Data/) contains example results for all nine subjects' training datasets.

## Time Majority Voting Algorithm for TCR and RWT dataset

Run [TMV_tcr_rwt.ipynb](TMV_tcr_rwt.ipynb) to execute the TMV algorithm on the TCR and the RWT dataset, and all of the results/graphs will be stored in [Time_Majority_results](Time_Majority_results). Currently, [Time_Majority_results/tcr](Time_Majority_results/tcr/) has example results generated for the TCR dataset. 

Inside the [Time_Majority_results](Time_Majority_results) folder, remeber to create a folder named "rwt" and a folder named [tcr](Time_Majority_results/tcr/). Inside each of these folders, create a folder called "TMV" (such as [Time_Majority_results/tcr/TMV](Time_Majority_results/tcr/TMV/)). Change the data_source to be either 'rwt' or 'tcr'. You can also decide to run which subjects by adjusting the variables "subject_id_start" and "subject_id_end". You can uncomment and comment out the corresponding names and classifiers in the second block to determine what classifiers you want to run. 

The TMV folder inside the "rwt" and "tcr" will only display one subject's result, and you can specify that subject by changing the variable "subject_id_search".

Please verify the increase of accuracy comparing to the Random Forest, and you procuced similar graphs in the paper. The link to the paper can be found on the [Research Tasks Forum](https://xiaodongqu.com/viewtopic.php?f=9&t=182&p=757#p757).

Side node: It took me around 10 - 20 minutes to finish running [TMV_tcr_rwt.ipynb](TMV_tcr_rwt.ipynb). Running time varies depending on different computers.
