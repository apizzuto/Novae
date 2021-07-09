## `random_forest/`

This directory contains all of the scripts used to train and use a model (RandomForest) to estimate the uncertainties on GRECO events. These uncertainties are the default (as of June 2021) for the `angErr` / `sigma` values in the GRECO dataset.

The model is trained on numu only, both NC and CC events, and is used to estimate the uncertainties for all flavors of monte carlo and all data events. 

### Instructions

To train a new model (if you have new MC), you will first need to add some columns to the data that the random forest needs, first, run 
```
python add_columns_for_training.py --infile=/path/to/mc_file
```

You are now ready to train a model. Run
```
python angularError_randomForest.py --infile=/path/to/mc_file_with_delta_psi.npy --minsamp=min_samples_split --log --boot
```

The commands `--minsamp=min_samples_split --log --boot` refer to hyperparameters of the model, you can read more about these in the script's argparser help. When you run this script, it will perform a grid search over a variety of hyperparameters and save that to a file. You can then read in the results from this training using `load_model.load_model()`.

If you are interested in running the hyperparameter scan over a variety of inputs, see the script `pycondor_grid_search.py` which details which parameters we scanned over initially in finding the best model.