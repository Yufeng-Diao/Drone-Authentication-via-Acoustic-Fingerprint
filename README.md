# Drone Authentication via Acoustic Fingerprint
The dataset will be released soon.

# Folder Description
## dataset_build
Build MFCC dataset based on collected drone audio. The format of the generated dataset is ".pkl".
## experiment
The code of all experiments, which are mentioned in the paper.
## runners
These files help with the training and evaluation process.
## toolbox
Some modules can be used in different programs.
## Detailed instruction will be completed ASAP

# Explain of Config file
- `originData_path`: The root of all drone audio.
- `output_path`: Where to output/obtain the model.
- `csv_savePath`: Where to save the evaluation results.
- `pkl_savePath`: Where to save the dataset of extracted MFCC features.

# At the Beginning
Download the drone audio dataset, then change all `originData_path` in all config files to the root of download drone audio.

# Experiment of Frame Length Change
1. Run `./dataset_build/pkl_gen_timeVar.py` to generate the dataset in `.pkl` format. This dataset is created from **DS1**.
   - Default dataset storage path is `./pkl_dataset/1_timeVar`.
2. Run `./experiment/timeVar/train_all_model.py` to train 8 different ML models on all generated dataset.
   - Default model storage path is `./trained_model/1_timeVar`.
3. Run `./experiment/timeVar/eval_all_model.py` to obtain the accuracy of each model on testset.
   - Default csv storage path is `./result/1_timeVar`.

# Filter-varying Experiment
## Overview
There are 9 config files for training different models.
- `config_filterVar_1_oneThird`
- `config_filterVar_1_twoThirds`
- `config_filterVar_1_all`
- `config_filterVar_2_oneThird`
- `config_filterVar_2_twoThirds`
- `config_filterVar_2_all`
- `config_filterVar_3_oneThird`
- `config_filterVar_3_twoThirds`
- `config_filterVar_3_all`
## Training and Evaluation
1. Run `./dataset_build/pkl_gen_filterVar.py` to generate the dataset in `.pkl` format. This dataset is created from **DS1**.
2. Change the name of config file in line 90 of `.\experiment\filterVar\train_all_model_filterVar.py` to train different models.
3. Change the name of config file in line 104 of `.\experiment\filterVar\eval_all_model_filterVar.py` to evaluate different models.
4. Run `.\experiment\filterVar\train_all_model_filterVar.py`.
   - Default model storage path is `./trained_model/2_filterVar/8d_x_xxxx`.
5. Run `.\experiment\filterVar\eval_all_model_filterVar.py`.
   - Default csv storage path is `./result/2_filterVar/8d_x_xxxx`.

# Filter-varying Experiment with AWGN
## Training and Evaluation
1. Run `./dataset_build/pkl_gen_filterVar_noise.py` to generate the dataset in `.pkl` format. This dataset is created from **DS1N**.
2. Change the name of config file in line 107 of `.\experiment\filterVar\eval_all_model_filterVar_noise.py` to evaluate different models.
3. Run `.\experiment\filterVar\eval_all_model_filterVar_noise.py`.
   - Default csv storage path is `./result/3_filterVar_noise/8d_x_xxxx`.

# Authentication of 24 drones without AWGN
## Training and Evaluation
1. Run `./experiment/noiseVar/train_all_model_noNoise.py`.
   - Default model storage path is `./trained_model/4_noiseVar/noNoise`.
2. Run `./experiment/noiseVar/eval_all_model_noNoise.py`.
   - The results will be shown in the console.



