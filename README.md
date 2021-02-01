# Beyond-Accuracy Reduction and Bias Amplification in Adversarial Personalized Ranking
This file presents the reproducibility details of the paper. **Beyond-Accuracy Reduction and Bias Amplification in Adversarial Personalized Ranking** submitted at SIGIR 2021 (Submission-id: 135)

**Table of Contents:**
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Reproducibility Details](#reproducibility-details)
  - [Image classification and feature extraction](#1-image-classification-and-feature-extraction)
  - [Recommendations generation](#2-recommendations-generation)
  - [Visual attacks](#3-visual-attacks)
  - [Recommendations generation after attack](#4-recommendations-generation-after-attack)
  - [Attack Success Rate and Feature Loss](#5-attack-success-rate-and-feature-loss)
  - [EXTRA: script input parameters](#extra-script-input-parameters)


## Requirements

To run the experiments, it is necessary to install the following requirements. 

* Python 3.6.9
* CUDA 10.1
* cuDNN 7.6.4

After having clone this repository with 
```
git clone repo-name
```
we suggest creating e virtual environment install the required Python dependencies with the following commands
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Datasets
The tested datasets across the paper experiments are reported in the following table.

|       Dataset      |   # Users   | # Products   |  # Feedback   | Density | *p(i|I<sub>SH</sub>)* | *p(i|I<sub>LT</sub>)* |
| ------------------ | ----------- | ------------ | ------------- | --------| --------------------- | --------------------- | 
|     Amazon Men     |    934      | 1,682        |  99,999       | 0.00630 | 0.6452                | 0.3548                 |


## Reproducibility Details

### 1. Training of the BPR-MF models.
The first step is to train the 
```
python classify_extract.py \
  --dataset <dataset_name> \
  --defense 0 \
  --gpu <gpu_id>
```
If you want to classify images with a defended model (i.e., Adversarial Training or Free Adversarial Training), run the following
```
python classify_extract.py \
  --dataset <dataset_name> \
  --gpu <gpu_id> \
  --defense 1 \
  --model_dir <model_name> \
  --model_file <model_filename>
```
This will produce ```classes.csv``` and ```features.npy```, which is a ```N X 2048``` float32 array corresponding to the extracted features for all ```N``` images. The two files are saved to ```./data/<dataset_name>/original/``` when no defense is applied, otherwise they are saved to ```./data/<dataset_name>/<model_name>_original/```. 

### 2. Recommendations generation
After this initial step, run the following command to train one of the available recommender models based on the extracted visual features:

```
python rec_generator.py \
  --dataset <dataset_name> \
  --gpu <gpu_id> \
  --experiment_name <full_experiment_name> \
  --epoch <num_training_epochs> \
  --verbose <show_results_each_n_epochs> \
  --topk 150
```
The recommeder models will be stored in ```./rec_model_weights/<dataset_name>/``` and the top-150 recommendation lists for each users will be saved in ```./rec_results/<dataset_name>/```. 

Extract the proposed rank-based metrics (CHR@K and nCDCG@K) you can execute the following command:
```
python evaluate_rec.py \
  --dataset <dataset_name> \
  --metric <ncdcg or chr> \
  --experiment_name <full_experiment_name> \
  -- origin <original_class_id> \
  --topk 150 \
  --analyzed_k <metric_k>
```

Results will be stored in ```./chr/<dataset_name>/``` and ```./ncdcg/<dataset_name>/``` in ```.tsv``` format. At this point, you can select from the extracted category-based metrics the origin-target pair of ids to execute the explored VAR attack scenario.

### 3. Visual attacks
**\*\*IMPORTANT\*\*** Apparently, there is a recent issue about CleverHans not being compatible with Tensorflow Addons (see [here](https://stackoverflow.com/questions/63896793/cleverhans-is-incompatible-with-tensorflow-addons)). As there is one script within CleverHans (i.e., ```cleverhans/attacks/spsa.py```) which imports tensorflow addons, a very rough (but effective) way to make it run is to comment out that line of code. More sophisticated solutions will be proposed in the future. <br>
Based upon the produced recommendation lists, choose an **origin** and a **target** class for each dataset. Then, run one of the available **targeted** attacks:
```
python classify_extract_attack.py \
  --dataset <dataset_name>  \
  --attack_type <attack_name> \
  [ATTACK_PARAMETERS] \
  --origin_class <zero_indexed_origin_class> \
  --target_class <zero_indexed_target_class> \
  --defense 0 \
  --gpu <gpu_id>
```
If you want to run an attack on a defended model, this is the command to run:
```
python classify_extract_attack.py \
  --dataset <dataset_name> \
  --attack_type <attack_name> \
  [ATTACK_PARAMETERS] \
  --origin_class <zero_indexed_origin_class> \
  --target_class <zero_indexed_target_class> \
  --defense 1 \
  --model_dir <model_name> \
  --model_file <model_filename> \
  --gpu <gpu_id>
```
This will produce (i) all attacked images, saved in ```tiff``` format to ```./data/<dataset_name>/<full_experiment_name>/images/``` and (ii) ```classes.csv``` and ```features.npy```. 

### 4. Recommendations generation after attack
Generate the recommendation lists for the produced visual attacks as specified in [Recommendations generation](#2-recommendations-generation).

### 5. Attack Success Rate and Feature Loss
In order to generate the attack Success Rate (SR) for each attack/defense combination, run the following script:
```
python -u evaluate_attack.py [SAME PARAMETERS SEEN FOR classify_extract_attack.py]
```
this will produce the text file ```./data/<dataset_name>/<full_experiment_name>/success_results.txt```, which contains the average SR results.

Then, to generate the Feature Loss (FL) for each attack/defense combination, run the following script:
```
python -u feature_loss.py [SAME PARAMETERS SEEN FOR classify_extract_attack.py]
```
this will generate the text file ```./data/<dataset_name>/full_experiment_name>/features_dist_avg_all_attack.txt``` with the average FL results, and the csv file ```./data/<dataset_name>/<full_experiment_name>/features_dist_all_attack.csv``` with the FL results for each attacked image. 

### EXTRA: script input parameters
```
# Possible values for 'dataset', 'defense', 'model_dir', 'model_file'
--dataset: {
    amazon_men 
    amazon_women
    tradesy
}

--defense : {
    0 # non-defended model
    1 # defended model
}

--model_dir : {
    madry # free adversarial training
    free_adv # free adversarial training
}

--model_file : {
    imagenet_linf_4.pt # adversarial training model filename
    model_best.pth.tar # free adversarial_training model filename
}

----------------------------------------------------------------------------------

# [ATTACK_PARAMETERS]
--eps # epsilon
--l # L norm

# All other attack parameters are hard-coded. Can be set as input parameters, too.
```

