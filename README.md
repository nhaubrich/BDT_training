# BDT_training
## Configure and train boosted decision tree classifiers on skimmed event ntuples
See `env.sh` for configuring python environment on lxplus. Specify training details in configuration in `config.py`, then train with `python train_bdt.py`. Estimate per-process scale factor uncertainties with a maximum likelihood fit in `combine.py`, currently configured with Barlow-Beeston-lite per-bin uncertainties and an inclusive luminosity uncertainty.
