# Drone Authentication via Acoustic Fingerprint
The dataset will be released soon.
# Experiment of Frame Length Change
1. Run `./dataset_build/pkl_gen_timeVar.py` to generate the dataset in `.pkl` format.
   - Default dataset storage path is `./pkl_dataset/1_timeVar`
2. Run `./experiment/timeVar/train_all_model.py` to train 8 different ML models on all generated dataset.
   - Default model storage path is `./trained_model/1_timeVar`
3. Run `./experiment/timeVar/eval_all_model.py` to obtain the accuracy of each model on testset.
   - Default csv storage path is `./result/1_timeVar`
4. 
## dataset_build
Build MFCC dataset based on collected drone audio. The format of the generated dataset is ".pkl".
## experiment
The code of all experiments, which are mentioned in the paper.
## runners
These files help with the training and evaluation process.
## toolbox
Some modules can be used in different programs.
## Detailed instruction will be completed ASAP
