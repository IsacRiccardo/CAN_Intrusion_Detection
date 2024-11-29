# CAN_Intrusion_Detection
Project to detect possible intrusions (Replay, Fuzzying, Combined) on the CAN bus using machine learning

# Datasets Description
Each row (used in the classification of the current CAN frame) contains the following information: the ID of the CAN frame, the IDs of the previous 4 CAN frames, and the data field of the current CAN frame. In addition, the last byte on each row indicates if the current frame is legitim (value 0) or attack (value 1).
* All datasets will be split into 70% training and 30% testing. (Can be modified inside the GLOBAL_VAR file)
## Folder structure
|------ Datasets \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Fuzzing \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle A \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle B \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle C \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Replay \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle A \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle B \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle C \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Combined \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle A \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle B \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------ Vehicle C 

# Classifiers used
* Random Forest
* XGB
* Logistic Regression
* SVM
* KNearest Neighbor

# Results
The results are automatically stored into the Reports folder created during execution. Each dataset will have its own report containing information about all the classifiers used like: Confusion Matrix and Accuracy.

# Tuning
The GLOBAL_VAR file will have parameters that will determine:
* The datasets folder
* The reports folder
* The model folder
* If reports are enabled
* If the Confusion Matrix will be plotted after every classifier
* If the model trained will be saved for further usage

# Execution

## Prerequesites
* Python installed

## Start execution
Execute run.bat to start

