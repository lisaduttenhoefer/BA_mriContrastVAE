#!/bin/bash

echo "=== Starting HC Training Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""

if [ -f "/workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py" ]; then
    echo "✓ Python script found"
else
    echo "✗ Python script NOT found"
    exit 1
fi

# # Vgm	neuromorphometrics	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm 

# # Vgm, Vwm, Vcsf	neuromorphometrics	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm Vwm Vcsf

# # Vgm	cobra	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm

# # Vgm, Vwm, Vcsf	cobra	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm Vwm Vcsf

# # Vgm	lpba40	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm

# # Vgm, Vwm, Vcsf	lpba40	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm Vwm Vcsf

# # Vgm	all	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm

# # Vgm, Vwm, Vcsf	all	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm Vwm Vcsf

# # Vgm	neuromorphometrics cobra lpba40	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm Vwm Vcsf

# # Vgm	neuromorphometrics cobra 	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra 	HC
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm Vwm Vcsf

# # Vgm	neuromorphometrics	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	neuromorphometrics	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# # Vgm	cobra	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	cobra	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# # Vgm	lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# # Vgm	all	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	all	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# # Vgm	neuromorphometrics cobra lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# # Vgm	neuromorphometrics cobra 	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm --norm_diagnosis "SCHZ"

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra 	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm Vwm Vcsf --norm_diagnosis "SCHZ"

# Vgm	neuromorphometrics	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	neuromorphometrics	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"

# Vgm	cobra	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	cobra	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name cobra --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"

# Vgm	lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name lpba40 --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"

# Vgm	all	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	all	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name all --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"

# Vgm	neuromorphometrics cobra lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra lpba40 --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"

# Vgm	neuromorphometrics cobra 	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm --norm_diagnosis "MDD"

# Vgm, Vwm, Vcsf	neuromorphometrics cobra 	MDD
python /workspace/project/catatonia_VAE-main_bq/src/run_ConVAE_2D_train_adapt.py --atlas_name neuromorphometrics cobra --volume_type Vgm Vwm Vcsf --norm_diagnosis "MDD"