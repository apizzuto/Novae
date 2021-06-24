## `random_forest/`

This directory contains all of the scripts used to train and use a model (RandomForest) to estimate the uncertainties on GRECO events. These uncertainties are the default (as of June 2021) for the `angErr` / `sigma` values in the GRECO dataset.

The model is trained on numu only, both NC and CC events, and is used to estimate the uncertainties for all flavors of monte carlo and all data events. 