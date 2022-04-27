# Erroneous_data_detection

Python3 - Version 3.7

Libraries Used for the project are:
1) pandas 
2) tqdm 
3) numpy 
4) datetime
5) sklearn
6) imblearn
7) scipy
8) tensorflow 
9) os

Project Flow:
1) All the preprocessing steps like one hot encoding, Normalization and feature engineering are performed in "PreProcessing.ipynb". This will generate two csv files, Train_updated.csv and Test_updated.csv
2) Now all the model (Decision Tree, Random Forest, Logistic Regression) training and hyperparameter tuning is performed in the file, "ModelRandomSearch.ipynb" file, this also generates a prediction.csv file which was submitted as part of the competition.
3) A ANN Model is trained and later tuned in the file, "NeuralNetwork.py".


Commands to Run:
1) After Installing all the libraries mentioned in the above list.
2) Run the Jupiter Notebooks mentioned above in the respective order.
3) Also, Run the python file NeuralNetwork.py using `python3 NeuralNetwork.py`
