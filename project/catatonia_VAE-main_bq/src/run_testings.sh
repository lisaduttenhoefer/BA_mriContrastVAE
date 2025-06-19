#!/bin/bash

echo "=== Starting HC Training Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""

if [ -f "/workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py" ]; then
    echo "✓ Python script found"
else
    echo "✗ Python script NOT found"
    exit 1
fi

# Vgm	neuromorphometrics	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_neuromorphometrics_20250605_1037

# Vgm, Vwm, Vcsf	neuromorphometrics	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_neuromorphometrics_20250530_1143

# # Vgm	cobra	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_cobra_20250601_2109

# # Vgm, Vwm, Vcsf	cobra	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_a_l_l_cobra_20250601_2109

# Vgm	lpba40	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_lpba40_20250605_2037

# # Vgm, Vwm, Vcsf	lpba40	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_lpba40_20250605_2235

# Vgm	all	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_all_20250601_2108

# # Vgm, Vwm, Vcsf	all	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_all_20250530_1142

# Vgm	neuromorphometrics cobra lpba40	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_neuromorphometrics_cobra_lpba40_20250606_0349

# Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	HC
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_lpba40_20250606_0540

# # Vgm	neuromorphometrics cobra 	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_cobra_neuromorphometrics_20250601_2108

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra 	HC
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_cobra_neuromorphometrics_20250530_1142

# # Vgm	neuromorphometrics	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_neuromorphometrics_20250605_1039

# # Vgm, Vwm, Vcsf	neuromorphometrics	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_neuromorphometrics_20250605_1135

# # Vgm	cobra	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_cobra_20250605_1236

# # Vgm, Vwm, Vcsf	cobra	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_cobra_20250605_1349

# # Vgm	lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_lpba40_20250605_1539

# # Vgm, Vwm, Vcsf	lpba40	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_lpba40_20250605_1729

# # Vgm	all	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_all_20250605_1850

# # Vgm, Vwm, Vcsf	all	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_all_20250605_1955

# Vgm	neuromorphometrics cobra lpba40	SCHZ
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_neuromorphometrics_cobra_lpba40_20250605_2051

# Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	SCHZ
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_lpba40_20250605_2148

# # Vgm	neuromorphometrics cobra 	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_V_g_m_neuromorphometrics_cobra_20250605_2244

# # Vgm, Vwm, Vcsf	neuromorphometrics cobra 	SCHZ
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_SCHZ_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_20250605_2340


# # Vgm	neuromorphometrics	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_neuromorphometrics_20250612_0940

# # Vgm, Vwm, Vcsf	neuromorphometrics	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_neuromorphometrics_20250612_1001

# # # Vgm	cobra	MDD
# # python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_cobra_20250605_1236

# # # Vgm, Vwm, Vcsf	cobra	MDD
# # python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_cobra_20250605_1349

# # Vgm	lpba40	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_lpba40_20250612_1103

# # Vgm, Vwm, Vcsf	lpba40	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_lpba40_20250612_1124

# # Vgm	all	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_all_20250612_1145

# # Vgm, Vwm, Vcsf	all	MDD
# python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_all_20250612_1209

# Vgm	neuromorphometrics cobra lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_neuromorphometrics_cobra_lpba40_20250612_1232

# Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	MDD
python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_lpba40_20250612_1252

# # # Vgm	neuromorphometrics cobra 	MDD
# # python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_V_g_m_neuromorphometrics_cobra_20250605_2244

# # # Vgm, Vwm, Vcsf	neuromorphometrics cobra 	MDD
# # python /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_MDD_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_20250605_2340
