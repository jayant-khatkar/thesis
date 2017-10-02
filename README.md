# Thesis Code

Hierarchal classification of benthoz organisms. (in progress) 


#### How to train a new model
In train_new.py:
1. create a list of the hyperparameters (need to change list to dictionary)
2. change the run_ID to a new number
3. in the train_model function, change the model loaded, and change the last argument to the hyperparameter list created
4. run `nohup python train_new.py > outputfile.log &`

The model is automatically saved to:
  ./<model_name>/<run_ID>/model_save

A tensorboard summary is automatically saved to:
  ./<model_name>/<run_ID>/tensorboard_log

The program will print dev and train performance at the end of each epoch to:
  outputfile.log

#### How to train a restored model_name
In train_restored.py:
1. change paths at the bottom (train_path/dev_path etc)
2. change model_name and run_ID
3. change save_path to whatever model to restore  
4. run  `nohup python train_restored.py > outputfile.log &`

outputs are the same as training a new model


#### How to test a restored model
In restorer.py:
1. change save_path to the location of the saved model and test path to the location of the test data
2. run  `python restorer`

it will print on screen the perofrmance of the model on the test data


#### Creating a new model
1. copy an existing model
2. the the prediction attribute function to be the structure of the new model
3. if different hyperparameters are used, change them in the load_hyperparameters attribute
4. change the name of the class and the model.name attribute
