# Drone Authentication via Acoustic Fingerprint
- [x] Basic tutorial about this project.
- [ ] Platform specification.
- [x] Package requirements.
- [ ] Link of drone audio dataset.

The dataset will be released soon.

# requirements
- matplotlib=3.5.0
- numpy=1.21.2
- pandas=1.3.5
- python=3.8.12
- seaborn=0.11.2
- wave=0.0.2
- yaml=0.2.5
- scikit-learn=1.0.2
- scipy=1.7.3

You need about 80G of storage space to generate the PKL dataset and models.

# Audio Dataset Explanation
- **DS1:** this dataset contained drone audio from No. 1 - No. 8.
- **DS:** this dataset contained drone audio from No. 1 - No. 24.
- **DS1N:** we added AWGN to ** DS1** with 0 dB SNR to create **DS1N**. The lengths of the corresponding drone audio in **DS1N** and **DS1** are equal to each other.
- **DS2N:** we added AWGN to **DS2** with 93 levels of \ac{snr} ranging from -8.00 dB to 15.00 dB with a step of 0.25 dB to create **DS2N**. The size of **DS2N** is 93 times larger than the size of **DS2**. In other words, each level of SNR creates a new subset in **DS2N**, whose size is equal to **DS2**.

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
2. Run `./experiment/timeVar/train_all_model.py` to train 8 different ML models on all generated datasets.
   - Default model storage path is `./trained_model/1_timeVar`.
3. Run `./experiment/timeVar/eval_all_model.py` to obtain the accuracy of each model on test set.
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
## PKL Generation
1. Run `./dataset_build/pkl_gen_filterVar.py` to generate the dataset in `.pkl` format. This dataset is created from **DS1**.
## Training and Evaluation
1. Change the name of config file in line 90 of `./experiment/filterVar/train_all_model_filterVar.py` to train different models.
2. Change the name of config file in line 104 of `./experiment/filterVar/eval_all_model_filterVar.py` to evaluate different models.
3. Run `./experiment/filterVar/train_all_model_filterVar.py`.
   - Default model storage path is `./trained_model/2_filterVar/8d_x_xxxx`.
4. Run `./experiment/filterVar/eval_all_model_filterVar.py`.
   - Default csv storage path is `./result/2_filterVar/8d_x_xxxx`.

# Filter-varying Experiment with AWGN
## PKL Generation
1. Run `./dataset_build/pkl_gen_filterVar_noise.py` to generate the dataset in `.pkl` format. This dataset is created from **DS1N**.
## Evaluation
1. Change the name of config file in line 107 of `./experiment/filterVar/eval_all_model_filterVar_noise.py` to evaluate different models.
2. Run `./experiment/filterVar/eval_all_model_filterVar_noise.py`.
   - Default csv storage path is `./result/3_filterVar_noise/8d_x_xxxx`.

# Authentication of 24 drones without AWGN
This experiment are conducted on **DS2**.
## Training and Evaluation
1. Run `./experiment/noiseVar/train_all_model_noNoise.py`.
   - Default model storage path is `./trained_model/4_noiseVar/noNoise`.
2. Run `./experiment/noiseVar/eval_all_model_noNoise.py`.
   - The results will be shown in the console.

# Authentication of 24 drones with AWGN
This experiment are conducted on **DS2N**.
## PKL Generation
1. Run `./dataset_build/pkl_gen_noiseVar.py` to generate the dataset in `.pkl` format. This dataset is created from **DS2N**.
## Evaluation
1. Run `./experiment/noiseVar/eval_all_model_noiseVar.py`.
   - Default csv storage path is `./result/4_noiseVar`.

# Security Study (Attack)
1. Run `./dataset_build/pkl_gen_base.py` to generate the dataset in `.pkl` format. This dataset is created from **DS2**.
2. Run `./experiment/attack/train_attack.py`.
3. Record `dic_reg`, `dic_attack`, `dic_bg` shown in console.
   - For example
   ![This is an image](./attack_example_1.png)
4. Change the value of `args.bg_type` and `args.attack_type` in `./experiment/attack/evaluate_attack.py` to `dic_bg` and `dic_attack`, respectively.
   - For example
   ![This is an image](./attack_example_2.png)
5. Run `./experiment/attack/evaluate_attack.py`.
   - The results will be shown in console.
   

