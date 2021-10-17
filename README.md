# Gransformer

The file then should be run for train and test is main.py.
Determine the dataset in config.py. 
You can change the configurations in the corresponding file in config_files directory.

For generation, execute "python main.py -g NUM" where NUM is an epoch number for which the trained model has been saved in model_save directory e.g., 3000.

For evaluation, use the file evaluate.py. Do not forget to set the correct dataset in config.py when running evaluate.py.
