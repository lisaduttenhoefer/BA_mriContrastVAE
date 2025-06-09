#!/bin/bash
echo "=== Starting HC Training Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""

# Cache löschen
echo "Clearing Python cache..."
find /workspace/project/catatonia_VAE-main_bq -name "*.pyc" -delete 2>/dev/null
find /workspace/project/catatonia_VAE-main_bq -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

if [ -f "/workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py" ]; then
echo "✓ Python script found"
else
echo "✗ Python script NOT found"
exit 1
fi

# Vgm	neuromorphometrics	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_neuromorphometrics_20250605_1037

# Vgm, Vwm, Vcsf	neuromorphometrics	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_neuromorphometrics_20250530_1143

# Vgm	cobra	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_cobra_20250601_2109

# Vgm, Vwm, Vcsf	cobra	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_a_l_l_cobra_20250601_2109

# Vgm	lpba40	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_lpba40_20250605_2037

# Vgm, Vwm, Vcsf	lpba40	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_lpba40_20250605_2235

# Vgm	all	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_all_20250601_2108

# Vgm, Vwm, Vcsf	all	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_all_20250530_1142

# Vgm	neuromorphometrics cobra lpba40	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_neuromorphometrics_cobra_lpba40_20250606_0349

# Vgm, Vwm, Vcsf	neuromorphometrics cobra lpba40	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_neuromorphometrics_cobra_lpba40_20250606_0540

# Vgm	neuromorphometrics cobra 	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_V_g_m_cobra_neuromorphometrics_20250601_2108

# Vgm, Vwm, Vcsf	neuromorphometrics cobra 	HC
python -B /workspace/project/catatonia_VAE-main_bq/src/RUN_ConVAE_2D_test_adapt.py --model_dir /workspace/project/catatonia_VAE-main_bq/analysis/TRAINING/norm_results_HC_Vgm_Vwm_Vcsf_cobra_neuromorphometrics_20250530_1142
