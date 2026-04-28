# Loan Approval Datset - COMP9417 Project

In order to run the code, follow this instruction:
1. Run the data split first, to create three .csv file: `train.csv`, `validation.csv`, `test.csv`. The command is
   
   `$ python data_split.py`

2. Next, run each of the following code to tune and train the respective model.
   - xRFM model: `$ python xRFM_model.py`
   - xGBoost model: `$ python xgboost_model.py`
   - MLP model: `$ python mlp_model.py`

    Note that these programs will also create `.joblib` to store the model, training time and best parameter.

3. In order to get the basic result such as training, inference time and accuracy as well as feature important analysis, run the command:

   `$ python statistic.py`

   Two `.csv` and multiple image would then be created.

4. Finally, to test the scalability of the models, run the command:
    
    `$ python training_time.py`