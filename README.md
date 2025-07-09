# MRI-based Normative Variational Autoencoder for Structural Brain Alterations in Psychiatric Disorders

This project employs a normative Variational Autoencoder (VAE) framework to objectively quantify and localize subtle structural brain deviations in major psychiatric disorders, offering interpretable insights into conditions like schizophrenia, major depressive disorder, and catatonia.

## Overview

This study addresses the critical need for objective and interpretable approaches in diagnosing and characterizing psychiatric illnesses. By leveraging a normative Variational Autoencoder (VAE) trained on healthy brain morphology, the project quantifies and localizes structural brain deviations in patients with schizophrenia (SCHZ), major depressive disorder (MDD), and catatonia (CTT). The VAE generates region-specific effect sizes, revealing distinct patterns of gray matter alterations and confirming its capability to identify diagnosis-specific anatomical deviations consistent with existing literature. Furthermore, the work explores correlations between effect sizes and psychiatric symptom severity, highlighting the potential for novel, objective biomarkers in neuropsychiatric disorders.

## Run Structure (step-by-step)

1.  **Set up the code structure** as described below.

2.  **Run training**: `"python run_ConVAE_2D_train_adapt.py"`
    * `--atlas_name` (Name of the desired atlas for training, default=`["all"]`)
    * `--volume_type` (Volume type(s) to use, default=`["Vgm", "Vwm", "Vcsf"]`)
    * `--num_epochs` (Number of epochs to be trained for, default=`200`)
    * `--n_bootstraps` (Number of bootstrap samples, default=`100`)
    * `--norm_diagnosis` (Which diagnosis is considered the "norm", default=`"HC"`)
    * `--train_ratio` (Norm split ratio, default=`0.7`)
    * `--batch_size` (Batch size, default=`16`)
    * `--learning_rate` (Learning rate, default=`0.000559`)
    * `--latent_dim` (Dimension of latent space, default=`20`)
    * `--kldiv_weight` (Weight for KL divergence loss, default=`1.2`)
    * `--save_models` (Save all bootstrap models, default=`True`)
    * `--no_cuda` (Disable CUDA (use CPU only), `action='store_true'`)
    * `--seed` (Random seed for reproducibility, default=`42`)
    * `--output_dir` (Override default output directory, default=`None`)
    * *This step will produce a folder in `"/project/catatonia_VAE-main_bq/analysis/TRAINING"` with all the necessary data needed for testing.*

3.  **Run testing**: `"python RUN_ConVAE_2D_test_adapt.py"`
    * `--model_dir` (Path to model directory, default: uses predefined path in code)
    * `& split_ctt = TRUE/FALSE` (Controls whether CTT-SCHZ and CTT-MDD are treated separately or combined as CTT)
    * *All information about the model to be used, atlas, volume type, etc., will be automatically loaded from the specified model directory.*

4.  **Output**:
    ```
    .
    └── [output_directory_name]/
        ├── figures/
        │   ├── distributions/
        │   │   ├── distribution plots for reconstruction loss, KL score and combined deviation score
        │   │   ├── errorbars global deviation scores for reconstruction loss, KL score and combined deviation score
        │   │   └── jitterplots for global deviation scores for reconstruction loss, KL score and combined deviation score
        │   ├── latent_embeddings/
        │   │   ├── UMAPs latent space colored for Co-Diagnosis, Dataset, Diagnosis and Sex
        │   ├── effect_size_distributions_vs_HC (distribution of regional effect sizes)
        │   ├── heatmap_1_ctt_regions_3diagnoses_vs_HC (heatmap effect sizes three diagnoses, top regions of CTT clinical group)
        │   ├── heatmap_2_overall_regions_3diagnoses_vs_HC (heatmap effect sizes three diagnoses, top regions of all clinical groups)
        │   ├── heatmap_3_ctt_subgroups_vs_HC (heatmap effect sizes three diagnoses, top regions of CTT clinical group, all subgroups divided by scores)
        │   ├── effect size error bar plots of diagnoses
        │   └── patient_correlations_fdr_bh_corrected
        └── all .csv files with the data from the plots

    ```

## Repo Structure (`src/`)

* **`models/`**: Model definitions for the contrast Variational Autoencoder.
    * **`base_model.py`**: Contains basic functions that the contrast VAEs can make use of.
    * **`ContrastVAE_2D.py`**: Code that defines and trains a Normative VAE. It includes the VAE model architecture, loss functions, and functions for training (including robust bootstrapping).

* **`module/`**: Functions to preprocess the feature maps for VAE training.
    * **`data_processing_hc.py`**: Code that primarily handles the loading, preprocessing, and splitting of MRI brain data from various atlases, preparing it for the VAE.

* **`utils/`**: Utility functions.
    * **`config_utils_model.py`**: Defines the configuration class that manages all parameters for the 2D VAE training run, including hyperparameters, data paths, and logging settings.
    * **`plotting_utils.py`**: Plotting and visualization utilities to display training metrics (loss curves), visualize the latent space, and analyze results from bootstrap training runs.
    * **`dev_scores_utils.py`**: Handles the calculation and visualization of deviation scores. It computes reconstruction errors and KL divergence for individual subjects, generates plots to illustrate these deviations, calculates regional effect sizes (Cliff's Delta) for specific brain regions, generates detailed heatmaps to visualize these regional effect sizes across diagnoses and datasets, and performs correlation analysis between deviation scores and clinical symptom severity scales.
    * **`logging_utils.py`**: Provides logging utilities for a VAE training session.
    * **`support_f.py`**: Contains utilities for data handling, splitting dataframes, and preprocessing.

## Data & Experiments (additional files needed for code usage)
* **`result_examples/`**: example results: file with all plots from testing (Neuromorphometrics/LPBA40, HC, Vgm) & html files for interactive plots and static brain plots
* **`data/`**: All data needed to run the code.
    * `.csv` and `.h5` files with the data of the patients.
    * `.nii` files for the atlases that you want to get visualized in the plot brains.

* **`data_training/`**: Already split metadata files (training & test) from the training step.
    * Generated by the training step automatically.
    * Needed to let testing run.

* **`metadata_20250110/`**: Complete metadata `.csv` files.
    * **`full_data_train_valid_test.csv`**: Complete metadata with age, sex, and diagnosis information.
    * **`full_data_with_codiagnosis_and_scores.csv`**: Complete metadata with CTT separated into CTT-SCHZ and CTT-MDD, and scores for ZI-files.
    * **`meta_data_NSS_all_variables.csv`** & **`meta_data_whiteCAT_all_variables.csv`**: Full metadata files of the whiteCAT and NSS datasets (with extended scores & values).

* **`analysis/`**: Results of the training and testing phases.
    * **`TESTING`**: All results of the testing run (all plots, sorted after run).
    * **`TRAINING`**: Trained models.

## Scripts & Notebooks (RUN files)

* **`RUN_ConVAE_2D_test_adapt.py`**: Testing code with adjustable parameters.
* **`run_ConVAE_2D_train_adapt.py`**: Training code with adjustable parameters.
* **`run_ConVAE_2D_tuning.py`**: Runs basic tuning for the normative VAE.
* **`run_testings.sh`**: Summarized runs for testings with different subsets of atlases, volume types, and norm diagnosis.
* **`run_trainings.sh`**: Summarized runs for trainings with different subsets of atlases, volume types, and norm diagnosis.
* **`saliency_maps.ipynb`**: Generates brain plots (plotting effect sizes on brain atlases); for static and `.html` for interactives.
* **`classification_tryout.ipynb`**: ROC analysis (logistic regression model), correlation between ROIs.

## Additional files

* **`preprocessing`**: All codes (parser & preprocessing) needed for preparing the `.xml` data.
