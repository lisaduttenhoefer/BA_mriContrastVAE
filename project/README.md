
# MRI-based Normative Variational Autoencoder for Structural Brain Alterations in Psychiatric Disorders
This project employs a normative Variational Autoencoder (VAE) framework to objectively quantify and localize subtle structural brain deviations in major psychiatric disorders, offering interpretable insights into conditions like schizophrenia, major depressive disorder, and catatonia.

## Overview
This study addresses the critical need for objective and interpretable approaches in diagnosing and characterizing psychiatric illnesses. By leveraging a normative Variational Autoencoder (VAE) trained on healthy brain morphology, the project quantifies and localizes structural brain deviations in patients with schizophrenia (SCHZ), major depressive disorder (MDD), and catatonia (CTT). The VAE generates region-specific effect sizes, revealing distinct patterns of gray matter alterations and confirming its capability to identify diagnosis-specific anatomical deviations consistent with existing literature. Furthermore, the work explores correlations between effect sizes and psychiatric symptom severity, highlighting the potential for novel, objective biomarkers in neuropsychiatric disorders.

##  Run Structure (step-by-step)

1. set up the code structure as described below 

2. run training with set parameters: "python run_ConVAE_2D_train_adapt.py"
    --atlas_name (Name of the desired atlas for training, default=["all"])
    --volume_type (Volume type(s) to use, default=["Vgm", "Vwm", "Vcsf"])
    --num_epochs (Number of epochs to be trained for, default=200)
    --n_bootstraps (Number of bootstrap samples, default=100)
    --norm_diagnosis (which diagnosis is considered the "norm", default="HC")
    --train_ratio (Normpslit ratio', default=0.7)
    --batch_size (Batch size, default=16)
    --learning_rate (Learning rate, default=0.000559) 
    --latent_dim (Dimension of latent space, default=20) 
    --kldiv_weight (Weight for KL divergence loss, default=1.2)  #vor tuning:4.0
    --save_models (Save all bootstrap models, default=True)
    --no_cuda (Disable CUDA (use CPU only), action='store_true')
    --seed (Random seed for reproducibility, default=42)
    --output_dir (Override default output directory, default=None)
        -> will produce a folder in "/project/catatonia_VAE-main_bq/analysis/TRAINING" with all the necessary data needed for testing

3. run testing "python RUN_ConVAE_2D_test_adapt.py"
    --model_dir (Path to model directory, default: uses predefined path in code)
    & split_ctt = TRUE/FALSE -> CTT-SCHZ and CTT-MDD or CTT
        -> all information about the model to be used, atlas, volume type etc. will be put into the Config function automatically from the given model directory

4. output:
    ├── figures
    │   ├── distributions
    │   │   ├── distribution plots for reconstruction loss, KL score and combined deviation score
    │   │   ├── errorbars global deviation scores for reconstruction loss, KL score and combined deviation score
    │   │   ├── jitterplors for global deviation scores for reconstruction loss, KL score and combined deviation score
    │   ├── latent_embeddings
    │   │   ├── UMAPs latent space colored for Co-Diagnosis, Dataset, Diagnosis and Sex
    │   ├── effect_size_distributions_vs_HC (distribution of regional effect sizes)
    │   ├── heatmap_1_ctt_regions_3diagnoses_vs_HC (heatmap effect sizes three diagnoses, top regions of CTT clinical group)
    │   ├── heatmap_2_overall_regions_3diagnoses_vs_HC (heatmap effect sizes three diagnoses, top regions of all clinical groups)
    │   ├── heatmap_3_ctt_subgroups_vs_HC (heatmap effect sizes three diagnoses, top regions of CTT clinical group, all subgroups divided by scores)
    │   ├── effect size error bar plots of diagnoses
    │   └── patient_correlations_fdr_bh_corrected
& all .csv files with the data from the plots

### Repo Structure (`src/`)

* `models/`: Model definitions, for the contrast Variational Autoencoder
                - `base_model.py`: basic functions the contrast VAEs can make use of
                - `ContrastVAE_2D.py`: code that defines and trains a Normative VAE. It includes the VAE model architecture, loss functions, and functions for training (including robust bootstrapping) 

* `module/`: functions to preprocess the feature maps for VAE training
                - `data_processing_hc.py`: code that primarily handles the loading, preprocessing, and splitting of MRI brain data from various atlases, preparing it for the VAE

* `utils/`: Utility functions 
                - `config_utils_model.py`: defines the configuration class that manages all parameters for the 2D VAE training run, including hyperparameters, data paths, and logging settings 
                - `plotting_utils.py`: plotting and visualization utilities, display training metrics (loss curves), visualize the latent space, and analyze results from bootstrap training runs
                - `dev_scores_utils.py`: calculation and visualization of deviation scores 
                        -> computes reconstruction errors and KL divergence for individual subjects
                        -> plots to illustrate these deviations
                        -> calculation of regional effect sizes (Cliff's Delta) for specific brain regions
                        -> generates detailed heatmaps to visualize these regional effect sizes across diagnoses and datasets
                        -> performs correlation analysis between deviation scores and clinical symptom severity scales
                - `logging_utils.py`: logging utilities for a VAE training session
                - `support_f.py`: utilities for data handling, splitting dfs and preprocessing
                        

### Data & Experiments (additional file needed for code usage)

* `data/`: all data needed to run the code
                -> .csv and .h5 files with the data of the patients 
                -> .nii files for the atlases that you want to get visualized in the plot brains

* `data_training/`: already split metadata files (training & test) from the training step 
                - generated by the training step automatically
                - needed to let testing run

* `metadata_20250110/`: metadata .csv files (complete)
                - `full_data_train_valid_test.csv`: complete metadata with age, sex and diagnosis information
                - `full_data_with_codiagnosis_and_scores.csv`: complete metadata with CTT separated in CTT-SCHZ and CTT-MDD & scores for ZI-files
                - `meta_data_NSS_all_variables.csv`&`meta_data_whiteCAT_all_variables.csv`: full metadata files of the whiteCAT and NSS datasets (with extended scores & values)

* `analysis/`: reseults of the training and testing phases
                - `TESTING`: all results of the testing run (all plots, sorted after run)
                - `TRAINING`: trained models 
                
### Scripts & Notebooks (RUN files)

* `RUN_ConVAE_2D_test_adapt.py`: testing code with adjustable parameters
* `run_ConVAE_2D_train_adapt.py`: training code with adjustable parameters
* `run_ConVAE_2D_tuning.py`: runs basic tuning for the normativ VAE
* `run_testings.sh`:summarized runs for testings with different subsets of atlases & volume types & norm diagnosis
* `run_trainings.sh`: summarized runs for trainings with different subsets of atlases & volume types & norm diagnosis
* `saliency_maps.ipynb`: generates brain plots (plotting effect sizes on brain atlases); 
                ->  for staticand .html for interactives
* `classification_tryout.ipynb`: ROC analysis (logistic regression model), Korrelation between ROIs

### Additional files

* `preprocessing`: all codes (parser & preprocessing) needed for preparing the .xml data 