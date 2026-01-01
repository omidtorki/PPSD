import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, ks_2samp, rankdata
import pandas as pd
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration ---
OUTPUT_DIR = Path("evaluation_plots_and_metrics_combined_v8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Real Data Paths
REAL_GENO_PATH = "geno_10000_2.txt" # Make sure this path is correct
REAL_PHENO_PATH = "pheno.txt"     # Make sure this path is correct

# Synthetic Data Paths
SYNTH_DATA_PATHS = {
    "GAN_Omid": {
        "geno": "geno_synth_omid.txt",
        "pheno": "pheno_synth_omid.txt",
        "num_snps": 10000
    },
    "GAN_Paper1": {
        "geno": "geno_synth_paper1.txt",
        "pheno": "pheno_synth_paper1.txt",
        "num_snps": 7160
    },
    "GAN_Paper2": {
        "geno": "geno_synth_paper2.txt",
        "pheno": "pheno_synth_paper2.txt",
        "num_snps": 10000
    }
}

# Original data characteristics
ORIGINAL_GENOTYPE_LEN = 10000
CHANNELS = 3 # for one-hot encoding if used (e.g. for discriminator)
NUM_CLASSES = 2

# Evaluation Parameters
MMD_SIGMA_LIST = [0.1, 1.0, 10.0]
PCA_N_COMPONENTS_EVAL = 2
PCA_SAMPLE_SIZE_EVAL = 5000 # Max samples to use for PCA plots from each dataset
DISCRIM_BATCH_SIZE_EVAL = 256 # If using discriminator evaluation
LD_WINDOW_FOR_PLOTS = 100 # Window for LD heatmaps and pairwise calculations

# Path to a pre-trained discriminator
# Set to None if not evaluating this or if model path isn't fixed/relevant
SAVED_DISCRIMINATOR_PATH = '../discriminator_epoch_30.keras' # Or None

# --- Helper Functions ---

def load_dosage_data(geno_path, expected_snps=None, label="Data"):
    print(f"Loading genotype data for {label} from {geno_path}...")
    if not os.path.exists(geno_path):
        raise FileNotFoundError(f"Genotype file not found: {geno_path}")
    dosages_float = np.loadtxt(geno_path, dtype=np.float32)

    if expected_snps is not None and dosages_float.shape[1] != expected_snps:
        print(f"Warning: {label} genotype file has {dosages_float.shape[1]} SNPs, expected {expected_snps}. Will use actual.")
    
    nan_mask = np.isnan(dosages_float)
    if np.any(nan_mask):
        print(f"Warning: Filling {np.sum(nan_mask)} NaN values with dosage 0.0 in {label} data.")
        dosages_float[nan_mask] = 0.0
    
    dosages_int = dosages_float.astype(np.int32)
    if np.min(dosages_int) < 0 or np.max(dosages_int) > 2:
        # Allow for empty arrays if a file is empty or problematic
        if dosages_int.size > 0:
             raise ValueError(f"{label} dosages should be 0, 1, or 2. Found min: {np.min(dosages_int)}, max: {np.max(dosages_int)}")
    print(f"{label} dosages loaded: {dosages_int.shape}")
    return dosages_int

def load_phenotype_data(label_path, label="Data"):
    print(f"Loading phenotype data for {label} from {label_path}...")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = np.loadtxt(label_path, dtype=int)
    print(f"{label} labels loaded: {labels.shape}")
    return labels

def load_keras_model(path, model_name="Model"):
    print(f"Loading {model_name} from {path}...")
    if not os.path.exists(path):
        print(f"Warning: Model file {path} not found for {model_name}.")
        return None
    custom_objects = {}
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Error loading {model_name} with compile=False: {e}. Trying with compile=True")
        try:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        except Exception as e2:
            print(f"Error loading {model_name} with compile=True either: {e2}")
            return None
    print(f"{model_name} loaded.")
    return model

def calculate_allele_frequencies_per_class(dosages, labels):
    n_snps = dosages.shape[1]
    af_controls = np.full(n_snps, np.nan); af_cases = np.full(n_snps, np.nan)
    control_indices = (labels == 0); case_indices = (labels == 1)
    if np.any(control_indices):
        dosages_controls = dosages[control_indices]; n_controls = dosages_controls.shape[0]
        if n_controls > 0: af_controls = (np.sum(dosages_controls == 1, axis=0) + 2 * np.sum(dosages_controls == 2, axis=0)) / (2 * n_controls)
    if np.any(case_indices):
        dosages_cases = dosages[case_indices]; n_cases = dosages_cases.shape[0]
        if n_cases > 0: af_cases = (np.sum(dosages_cases == 1, axis=0) + 2 * np.sum(dosages_cases == 2, axis=0)) / (2 * n_cases)
    return af_controls, af_cases

def calculate_chi_squared_np(genotype_dosages, labels):
    n_samples, n_snps = genotype_dosages.shape; chi_sq_values = np.zeros(n_snps, dtype=np.float32)
    is_case = (labels == 1); is_control = (labels == 0)
    cases_data = genotype_dosages[is_case]; controls_data = genotype_dosages[is_control]
    n_cases = cases_data.shape[0]; n_controls = controls_data.shape[0]
    if n_cases == 0 or n_controls == 0: return chi_sq_values # or np.full(n_snps, np.nan)
    for snp_idx in range(n_snps):
        obs_alt_ca = np.sum(cases_data[:, snp_idx] == 1) + 2 * np.sum(cases_data[:, snp_idx] == 2)
        obs_ref_ca = 2 * n_cases - obs_alt_ca
        obs_alt_co = np.sum(controls_data[:, snp_idx] == 1) + 2 * np.sum(controls_data[:, snp_idx] == 2)
        obs_ref_co = 2 * n_controls - obs_alt_co
        total_ref = obs_ref_ca + obs_ref_co; total_alt = obs_alt_ca + obs_alt_co
        total_alleles = 2 * (n_cases + n_controls)
        if total_alleles == 0 or total_ref == 0 or total_alt == 0: chi_sq_values[snp_idx] = 0.0; continue
        # Expected counts
        exp_ref_ca = (total_ref * (2 * n_cases)) / total_alleles
        exp_alt_ca = (total_alt * (2 * n_cases)) / total_alleles
        exp_ref_co = (total_ref * (2 * n_controls)) / total_alleles
        exp_alt_co = (total_alt * (2 * n_controls)) / total_alleles
        # Chi-squared terms
        term1 = ((obs_ref_ca - exp_ref_ca)**2) / (exp_ref_ca + 1e-9) # Add epsilon for stability
        term2 = ((obs_alt_ca - exp_alt_ca)**2) / (exp_alt_ca + 1e-9)
        term3 = ((obs_ref_co - exp_ref_co)**2) / (exp_ref_co + 1e-9)
        term4 = ((obs_alt_co - exp_alt_co)**2) / (exp_alt_co + 1e-9)
        chi_sq_values[snp_idx] = term1 + term2 + term3 + term4
    return np.nan_to_num(chi_sq_values)


def rank_transform_matrix(matrix):
    if matrix.ndim != 2: raise ValueError("Input must be a 2D matrix for rank transformation.")
    flat_matrix = matrix.flatten(); nan_mask = np.isnan(flat_matrix); valid_values = flat_matrix[~nan_mask]
    if valid_values.size == 0: return np.full_like(matrix, np.nan)
    ranks = rankdata(valid_values, method='dense'); min_rank = np.min(ranks); max_rank = np.max(ranks)
    if max_rank == min_rank: normalized_ranks = np.full_like(ranks, 0.5, dtype=float)
    else: normalized_ranks = (ranks - min_rank) / (max_rank - min_rank)
    ranked_flat_matrix = np.full_like(flat_matrix, np.nan, dtype=float)
    ranked_flat_matrix[~nan_mask] = normalized_ranks
    return ranked_flat_matrix.reshape(matrix.shape)

def calculate_ld_r2_np_for_eval(snp_data_dosage, window_size=None, dataset_name="Data"):
    if snp_data_dosage.ndim != 2 or snp_data_dosage.shape[0] < 2:
        print(f"   Warning: Invalid dosage data shape for LD calculation for {dataset_name}. Skipping."); return np.array([])
    n_samples, n_snps = snp_data_dosage.shape
    if window_size is None or window_size > n_snps: window_size = n_snps
    elif window_size <= 1:
        print(f"   Warning: Window size {window_size} is too small for LD for {dataset_name}. Using all {n_snps} SNPs or skipping if n_snps < 2.");
        if n_snps < 2: return np.array([])
        window_size = n_snps # Use all SNPs if window is too small but data has >=2 SNPs

    snp_data_window = snp_data_dosage[:, :window_size].astype(float)
    print(f"   Calculating LD matrix (r^2) for first {snp_data_window.shape[1]} SNPs of {dataset_name} ({n_samples} samples)...")
    start_ld_time = time.time()
    
    # Impute NaNs with mode for LD calculation if any 
    for i in range(snp_data_window.shape[1]):
        col = snp_data_window[:, i]
        if np.any(np.isnan(col)):
            valid_col = col[~np.isnan(col)]
            mode_val = np.bincount(valid_col.astype(int)).argmax() if valid_col.size > 0 else 0
            snp_data_window[np.isnan(col), i] = mode_val

    # Use pandas for robust correlation calculation, handling potential low variance columns
    df = pd.DataFrame(snp_data_window)
    # min_periods can be important if some SNPs have many missing values (not our case here after imputation)
    # or if sample size is very small. max(2, int(n_samples * 0.1)) is a heuristic.
    corr_matrix_r = df.corr(method='pearson', min_periods=max(2, int(n_samples * 0.05))).values 
    corr_matrix_r[np.isnan(corr_matrix_r)] = 0.0 # Replace NaN correlations (e.g. from zero-variance SNPs) with 0
    ld_matrix_r2 = np.square(corr_matrix_r)
    print(f"   Finished LD matrix calculation for {dataset_name} in {time.time() - start_ld_time:.2f} sec.")
    return ld_matrix_r2

# --- Combined Evaluation Functions ---

def plot_combined_allele_frequencies(dosages_real_full, labels_real, synth_datasets_info_list, output_dir):
    print("\n--- Evaluating Combined Allele Frequencies ---")
    metrics_results = {}
    num_gan = len(synth_datasets_info_list)

    for class_label, class_name_str in [(0, "Controls"), (1, "Cases")]:
        # --- Scatter Plots: Real AF vs Synthetic AF ---
        fig_scatter, axes_scatter = plt.subplots(1, num_gan, figsize=(5 * num_gan, 5), sharex=True, sharey=True)
        if num_gan == 1: axes_scatter = [axes_scatter] # Ensure axes_scatter is always iterable

        all_af_values_real_class = []
        all_af_values_synth_class = []

        # First pass: calculate all AFs and collect for global scaling
        temp_af_data = []
        for i, synth_info in enumerate(synth_datasets_info_list):
            name = synth_info['name']
            dosages_synth = synth_info['geno']
            labels_synth = synth_info['pheno']
            num_snps_synth = synth_info['num_snps']

            # Slice real data to match current synthetic data's SNP count for this comparison
            dosages_real_sliced = dosages_real_full[:, :num_snps_synth]
            
            af_ctrl_real, af_case_real = calculate_allele_frequencies_per_class(dosages_real_sliced, labels_real)
            af_ctrl_synth, af_case_synth = calculate_allele_frequencies_per_class(dosages_synth, labels_synth)

            af_real_current = af_ctrl_real if class_label == 0 else af_case_real
            af_synth_current = af_ctrl_synth if class_label == 0 else af_case_synth
            
            valid_indices = ~np.isnan(af_real_current) & ~np.isnan(af_synth_current)
            af_real_valid = af_real_current[valid_indices]
            af_synth_valid = af_synth_current[valid_indices]
            temp_af_data.append({'name': name, 'real': af_real_valid, 'synth': af_synth_valid})

            if af_real_valid.size > 0: all_af_values_real_class.extend(af_real_valid)
            if af_synth_valid.size > 0: all_af_values_synth_class.extend(af_synth_valid)
        
        # Determine global min/max for scatter plot axes
        global_min_af = min(np.min(all_af_values_real_class) if all_af_values_real_class else 0, 
                            np.min(all_af_values_synth_class) if all_af_values_synth_class else 0)
        global_max_af = max(np.max(all_af_values_real_class) if all_af_values_real_class else 1, 
                            np.max(all_af_values_synth_class) if all_af_values_synth_class else 1)
        padding_af = (global_max_af - global_min_af) * 0.05 # 5% padding

        # Second pass: plot with consistent scales
        for i, data in enumerate(temp_af_data):
            name = data['name']
            af_real_valid = data['real']
            af_synth_valid = data['synth']

            if af_real_valid.size == 0:
                print(f"No valid overlapping AFs for {name} - {class_name_str} to compare scatter.")
                axes_scatter[i].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                axes_scatter[i].set_title(f"{name}\nvs Real ({class_name_str})")
                continue

            axes_scatter[i].scatter(af_real_valid, af_synth_valid, alpha=0.3, s=10)
            axes_scatter[i].plot([global_min_af, global_max_af], [global_min_af, global_max_af], 'r--', label="Ideal y=x")
            axes_scatter[i].set_title(f"{name}\nvs Real ({class_name_str})")
            axes_scatter[i].grid(True)
            if i == 0: axes_scatter[i].set_ylabel(f"Synthetic Data Allele Frequency")
            axes_scatter[i].set_xlabel(f"Real Data Allele Frequency")
            axes_scatter[i].set_xlim(global_min_af - padding_af, global_max_af + padding_af)
            axes_scatter[i].set_ylim(global_min_af - padding_af, global_max_af + padding_af)
            if i == num_gan -1 : axes_scatter[i].legend(loc="lower right")

            mae = mean_absolute_error(af_real_valid, af_synth_valid)
            rmse = np.sqrt(mean_squared_error(af_real_valid, af_synth_valid))
            metrics_results[f"{name}_AF_MAE_{class_name_str}"] = mae
            metrics_results[f"{name}_AF_RMSE_{class_name_str}"] = rmse
            print(f"AF MAE ({name}, {class_name_str}): {mae:.4f}, RMSE: {rmse:.4f}")

        fig_scatter.suptitle(f"SNP Allele Frequency Comparison ({class_name_str})", fontsize=16)
        fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"af_scatter_combined_{class_name_str.lower()}.png")
        plt.close(fig_scatter)

        # --- Histograms of AF Distributions (Real vs. Each GAN) ---
        fig_hist, axes_hist = plt.subplots(1, num_gan, figsize=(6 * num_gan, 5), sharey=True)
        if num_gan == 1: axes_hist = [axes_hist]
        
        # Determine global x-axis range for histograms
        # temp_af_data already has the valid AFs for real and synth per GAN
        all_hist_af_values = []
        for data in temp_af_data:
            if data['real'].size > 0: all_hist_af_values.extend(data['real'])
            if data['synth'].size > 0: all_hist_af_values.extend(data['synth'])
        
        if not all_hist_af_values:
            print(f"No AF data to plot histograms for {class_name_str}.")
            plt.close(fig_hist) # Close empty figure
        else:
            hist_min_val = np.min(all_hist_af_values)
            hist_max_val = np.max(all_hist_af_values)
            common_bins = np.linspace(hist_min_val, hist_max_val, 51) # 50 bins

            for i, data in enumerate(temp_af_data):
                name = data['name']
                af_real_valid = data['real']
                af_synth_valid = data['synth']

                if af_real_valid.size == 0 and af_synth_valid.size == 0:
                    axes_hist[i].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                else:
                    sns.histplot(af_real_valid, color="blue", label="Real AFs", kde=False, stat="density", bins=common_bins, element="step", fill=False, ax=axes_hist[i])
                    sns.histplot(af_synth_valid, color="red", label=f"{name} AFs", kde=False, stat="density", bins=common_bins, element="step", fill=False, ax=axes_hist[i])
                axes_hist[i].set_title(f"AF Distribution: Real vs {name}\n({class_name_str})")
                axes_hist[i].set_xlabel(f"Allele Frequency")
                if i == 0: axes_hist[i].set_ylabel("Density")
                axes_hist[i].legend()
                axes_hist[i].grid(True)
            
            fig_hist.suptitle(f"Distribution of Allele Frequencies ({class_name_str})", fontsize=16)
            fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_dir / f"af_histograms_overlayed_combined_{class_name_str.lower()}.png")
            plt.close(fig_hist)
            
    return metrics_results

def plot_combined_chi_squared_distribution(dosages_real_full, labels_real, synth_datasets_info_list, output_dir):
    print("\n--- Evaluating Combined Chi-squared Statistic Distribution ---")
    metrics_results = {}
    num_gan = len(synth_datasets_info_list)

    # --- Histograms of Chi-squared Values ---
    fig_hist, axes_hist = plt.subplots(1, num_gan, figsize=(6 * num_gan, 5), sharey=True)
    if num_gan == 1: axes_hist = [axes_hist]

    all_chi_sq_values_for_scaling = []
    chi_sq_data_for_plotting = []

    # First pass: calculate all Chi-sq and collect for global scaling and KS/MMD
    for synth_info in synth_datasets_info_list:
        name = synth_info['name']
        dosages_synth = synth_info['geno']
        labels_synth = synth_info['pheno']
        num_snps_synth = synth_info['num_snps']

        dosages_real_sliced = dosages_real_full[:, :num_snps_synth]
        
        chi_sq_real = calculate_chi_squared_np(dosages_real_sliced, labels_real)
        chi_sq_synth = calculate_chi_squared_np(dosages_synth, labels_synth)

        chi_sq_real_valid = chi_sq_real[~np.isnan(chi_sq_real)]
        chi_sq_synth_valid = chi_sq_synth[~np.isnan(chi_sq_synth)]
        chi_sq_data_for_plotting.append({'name': name, 'real': chi_sq_real_valid, 'synth': chi_sq_synth_valid, 'real_full_snp_set': chi_sq_real, 'synth_full_snp_set': chi_sq_synth})

        if chi_sq_real_valid.size > 0: all_chi_sq_values_for_scaling.extend(chi_sq_real_valid)
        if chi_sq_synth_valid.size > 0: all_chi_sq_values_for_scaling.extend(chi_sq_synth_valid)

        if chi_sq_real_valid.size > 0 and chi_sq_synth_valid.size > 0:
            ks_statistic, ks_p_value = ks_2samp(chi_sq_real_valid, chi_sq_synth_valid)
            # MMD requires scipy.stats.gaussian_kde or similar for sigma estimation, or fixed sigma
            # mmd_chi_sq_eval = mmd_loss_np(chi_sq_real_valid[:, np.newaxis], chi_sq_synth_valid[:, np.newaxis], sigma_list=MMD_SIGMA_LIST) # if mmd_loss_np is defined
            # metrics_results[f"{name}_ChiSq_MMD_Eval"] = mmd_chi_sq_eval
            # print(f"MMD between Real and {name} Chi-squared: {mmd_chi_sq_eval:.4f}")
            print(f"{name} Chi-squared KS-test: Stat={ks_statistic:.4f}, p-val={ks_p_value:.4g}")
            metrics_results[f"{name}_ChiSq_KS_Stat"] = ks_statistic
            metrics_results[f"{name}_ChiSq_KS_PValue"] = ks_p_value

            valid_comparison_mask = ~np.isnan(chi_sq_real) & ~np.isnan(chi_sq_synth) # Use original arrays for pairwise
            if np.any(valid_comparison_mask):
                mae_chi_sq = mean_absolute_error(chi_sq_real[valid_comparison_mask], chi_sq_synth[valid_comparison_mask])
                rmse_chi_sq = np.sqrt(mean_squared_error(chi_sq_real[valid_comparison_mask], chi_sq_synth[valid_comparison_mask]))
                print(f"{name} Chi-squared MAE (pairwise): {mae_chi_sq:.4f}, RMSE: {rmse_chi_sq:.4f}")
                metrics_results[f"{name}_ChiSq_MAE_Pairwise"] = mae_chi_sq
                metrics_results[f"{name}_ChiSq_RMSE_Pairwise"] = rmse_chi_sq
            else:
                metrics_results[f"{name}_ChiSq_MAE_Pairwise"] = np.nan
                metrics_results[f"{name}_ChiSq_RMSE_Pairwise"] = np.nan
        else:
            print(f"Not enough valid ChiSq data for KS-test or MAE/RMSE for {name}.")
            metrics_results[f"{name}_ChiSq_KS_Stat"] = np.nan; metrics_results[f"{name}_ChiSq_KS_PValue"] = np.nan
            metrics_results[f"{name}_ChiSq_MAE_Pairwise"] = np.nan; metrics_results[f"{name}_ChiSq_RMSE_Pairwise"] = np.nan


    if not all_chi_sq_values_for_scaling:
        print("No Chi-squared data to plot histograms.")
        plt.close(fig_hist)
    else:
        hist_min_chi = np.min(all_chi_sq_values_for_scaling)
        hist_max_chi = np.max(all_chi_sq_values_for_scaling)
        # Avoid issues if max is very large due to outliers, cap for plotting or use log scale
        # For now, linear scale with common bins based on actual range.
        hist_max_chi = np.percentile(all_chi_sq_values_for_scaling, 99.5) # Cap at 99.5th percentile for better viz
        common_bins_chi = np.linspace(hist_min_chi, hist_max_chi, 76) # 75 bins

        # Second pass: plot histograms
        for i, data in enumerate(chi_sq_data_for_plotting):
            name = data['name']
            chi_sq_real_valid = data['real']
            chi_sq_synth_valid = data['synth']

            if chi_sq_real_valid.size == 0 and chi_sq_synth_valid.size == 0:
                 axes_hist[i].text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                sns.histplot(chi_sq_real_valid, color="blue", label="Real ChiSq", kde=True, stat="density", bins=common_bins_chi, element="step", fill=False, ax=axes_hist[i])
                sns.histplot(chi_sq_synth_valid, color="red", label=f"{name} ChiSq", kde=True, stat="density", bins=common_bins_chi, element="step", fill=False, ax=axes_hist[i])
            axes_hist[i].set_title(f"Real vs {name}")
            axes_hist[i].set_xlabel("Chi-squared Statistic Value")
            if i == 0: axes_hist[i].set_ylabel("Density")
            axes_hist[i].legend()
            axes_hist[i].grid(True)
            axes_hist[i].set_xlim(left=hist_min_chi, right=hist_max_chi) # Apply cap

        fig_hist.suptitle("Distribution of Chi-squared Association Statistics", fontsize=16)
        fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / "chi_squared_distribution_combined.png")
        plt.close(fig_hist)

    # --- Box Plots of Chi-squared Values ---
    fig_box, ax_box = plt.subplots(figsize=(max(8, 2 * (num_gan + 1)), 6)) # Make wider for more GANs
    plot_data_for_boxplot = []
    labels_boxplot = []

    # Real data (once, using full SNP set unless a smaller common set is preferred)
    # For boxplot, let's use the smallest common SNP set (GAN2's 7160) for the "Real" box
    # or show "Real (10k SNPs)" and "Real (7160 SNPs)" if very different
    num_snps_gan2 = SYNTH_DATA_PATHS["GAN_Paper1"]["num_snps"]
    chi_sq_real_common_subset = calculate_chi_squared_np(dosages_real_full[:, :num_snps_gan2], labels_real)
    chi_sq_real_common_valid = chi_sq_real_common_subset[~np.isnan(chi_sq_real_common_subset)]
    if chi_sq_real_common_valid.size > 0:
        plot_data_for_boxplot.append(chi_sq_real_common_valid)
        labels_boxplot.append(f"Real ({num_snps_gan2} SNPs)")
    
    for data in chi_sq_data_for_plotting:
        if data['synth'].size > 0:
            plot_data_for_boxplot.append(data['synth']) # synth is already valid and for its specific SNP set
            labels_boxplot.append(f"{data['name']} ({data['synth_full_snp_set'].shape[0]} SNPs)")
    
    if plot_data_for_boxplot:
        # Cap outliers for better boxplot visualization if ranges are huge
        plot_data_for_boxplot_capped = [d[d < np.percentile(d, 99.5)] if len(d) > 0 else d for d in plot_data_for_boxplot]

        sns.boxplot(data=plot_data_for_boxplot_capped, palette=["skyblue"] + ["salmon"] * num_gan, showfliers=True, ax=ax_box) # Showfliers can be False
        ax_box.set_xticklabels(labels_boxplot, rotation=15, ha="right")
        ax_box.set_ylabel("Chi-squared Value (capped at 99.5th percentile for viz)")
        ax_box.set_title("Box Plot of Chi-squared Values (Real vs. Synthetic GANs)")
        ax_box.grid(True, axis='y')
        fig_box.tight_layout()
        plt.savefig(output_dir / "chi_squared_boxplot_combined.png")
    else:
        print("   Not enough data for combined Box plot of Chi-squared values.")
    plt.close(fig_box)
            
    return metrics_results

def plot_combined_ld_structure(dosages_real_full, synth_datasets_info_list, output_dir, ld_window_config=LD_WINDOW_FOR_PLOTS):
    print(f"\n--- Evaluating Combined LD Structure (Configured Window: {ld_window_config} SNPs) ---")
    metrics_results = {}
    num_gan = len(synth_datasets_info_list)

    # Determine the smallest number of SNPs among all datasets to define a common window for comparison
    # This is important if ld_window_config is larger than what GAN2 offers
    min_snps_across_all = min(dosages_real_full.shape[1], 
                              min(s['num_snps'] for s in synth_datasets_info_list))
    
    actual_ld_window = min(ld_window_config, min_snps_across_all)
    if actual_ld_window < 2:
        print(f"   Actual LD window size {actual_ld_window} is too small for LD. Skipping LD plots.")
        return metrics_results
    print(f"   Using an actual LD window of {actual_ld_window} SNPs for all comparisons.")

    # --- LD Heatmaps (Original r^2 and Rank-Transformed) ---
    # One figure for Original r^2 (Real + 3 GANs), One for Rank-Transformed (Real + 3 GANs)
    # Or combine into a 2x(num_gan+1) or (num_gan+1)x2 grid if preferred. Let's do (num_gan+1) columns, 2 rows.
    
    fig_heatmap, axes_heatmap = plt.subplots(2, num_gan + 1, figsize=(5 * (num_gan + 1), 10), squeeze=False)

    # Calculate LD for Real data (sliced to common window)
    ld_matrix_real_r2 = calculate_ld_r2_np_for_eval(dosages_real_full[:, :actual_ld_window], 
                                                    window_size=actual_ld_window, dataset_name="Real (Common Window)")
    ld_matrix_real_ranked = rank_transform_matrix(ld_matrix_real_r2) if ld_matrix_real_r2.size > 0 else np.array([])

    # Plot Real LD
    if ld_matrix_real_r2.size > 0:
        im_real_orig = axes_heatmap[0, 0].imshow(ld_matrix_real_r2, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes_heatmap[0, 0].set_title(f'Real LD (r²)\n{actual_ld_window} SNPs')
        fig_heatmap.colorbar(im_real_orig, ax=axes_heatmap[0, 0], fraction=0.046, pad=0.04)
    else: axes_heatmap[0, 0].text(0.5,0.5, "No LD Data", ha='center', va='center')
    
    if ld_matrix_real_ranked.size > 0:
        im_real_rank = axes_heatmap[1, 0].imshow(ld_matrix_real_ranked, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axes_heatmap[1, 0].set_title(f'Rank-Transformed Real LD\n{actual_ld_window} SNPs')
        fig_heatmap.colorbar(im_real_rank, ax=axes_heatmap[1, 0], fraction=0.046, pad=0.04)
    else: axes_heatmap[1, 0].text(0.5,0.5, "No LD Data", ha='center', va='center')


    ld_matrices_synth_r2 = {}
    # Calculate and Plot LD for Synthetic data
    for i, synth_info in enumerate(synth_datasets_info_list):
        name = synth_info['name']
        dosages_synth = synth_info['geno']
        
        # Synthetic data already has its specific number of SNPs, slice to common window
        ld_matrix_synth_r2 = calculate_ld_r2_np_for_eval(dosages_synth[:, :actual_ld_window],
                                                         window_size=actual_ld_window, dataset_name=f"{name} (Common Window)")
        ld_matrices_synth_r2[name] = ld_matrix_synth_r2 # Store for later pairwise comparison
        ld_matrix_synth_ranked = rank_transform_matrix(ld_matrix_synth_r2) if ld_matrix_synth_r2.size > 0 else np.array([])

        ax_col = i + 1 # Column in subplot grid
        if ld_matrix_synth_r2.size > 0:
            im_synth_orig = axes_heatmap[0, ax_col].imshow(ld_matrix_synth_r2, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            axes_heatmap[0, ax_col].set_title(f'{name} LD (r²)\n{actual_ld_window} SNPs')
            fig_heatmap.colorbar(im_synth_orig, ax=axes_heatmap[0, ax_col], fraction=0.046, pad=0.04)
        else: axes_heatmap[0, ax_col].text(0.5,0.5, "No LD Data", ha='center', va='center')

        if ld_matrix_synth_ranked.size > 0:
            im_synth_rank = axes_heatmap[1, ax_col].imshow(ld_matrix_synth_ranked, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            axes_heatmap[1, ax_col].set_title(f'Rank-Transformed {name} LD\n{actual_ld_window} SNPs')
            fig_heatmap.colorbar(im_synth_rank, ax=axes_heatmap[1, ax_col], fraction=0.046, pad=0.04)
        else: axes_heatmap[1, ax_col].text(0.5,0.5, "No LD Data", ha='center', va='center')

    fig_heatmap.suptitle(f"LD Heatmaps (Original r² and Rank-Transformed) - First {actual_ld_window} SNPs", fontsize=16)
    fig_heatmap.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"ld_heatmaps_combined_ws{actual_ld_window}.png")
    plt.close(fig_heatmap)

    # --- Scatter plot of original r^2 values for pairs (Real vs Each GAN) ---
    if ld_matrix_real_r2.size > 0:
        fig_scatter_ld, axes_scatter_ld = plt.subplots(1, num_gan, figsize=(6 * num_gan, 5.5), sharex=True, sharey=True)
        if num_gan == 1: axes_scatter_ld = [axes_scatter_ld]

        iu_indices = np.triu_indices(actual_ld_window, k=1)
        ld_real_pairs = ld_matrix_real_r2[iu_indices]
        
        # Determine global min/max for scatter plot axes for LD values (0 to 1)
        # Scatter plots always from -0.05 to 1.05 for r^2
        scatter_xlim = (-0.05, 1.05)
        scatter_ylim = (-0.05, 1.05)

        for i, synth_info in enumerate(synth_datasets_info_list):
            name = synth_info['name']
            ld_matrix_synth_r2_current = ld_matrices_synth_r2.get(name)

            if ld_matrix_synth_r2_current is None or ld_matrix_synth_r2_current.size == 0:
                axes_scatter_ld[i].text(0.5, 0.5, 'No Synth LD Data', ha='center', va='center')
                axes_scatter_ld[i].set_title(f"Real vs {name}\nPairwise LD (r²)")
                continue

            ld_synth_pairs = ld_matrix_synth_r2_current[iu_indices]
            
            axes_scatter_ld[i].scatter(ld_real_pairs, ld_synth_pairs, alpha=0.2, s=10)
            axes_scatter_ld[i].plot([0, 1], [0, 1], 'r--', label="Ideal y=x") # Ideal line based on 0-1 range
            axes_scatter_ld[i].set_title(f"Real vs {name}\nPairwise LD (r²)")
            axes_scatter_ld[i].set_xlabel(f"Real Data LD (r²)")
            if i == 0: axes_scatter_ld[i].set_ylabel(f"Synthetic Data LD (r²)")
            axes_scatter_ld[i].grid(True)
            axes_scatter_ld[i].legend()
            axes_scatter_ld[i].set_xlim(scatter_xlim)
            axes_scatter_ld[i].set_ylim(scatter_ylim)

            if len(ld_real_pairs) > 1 and len(ld_synth_pairs) > 1: # Need at least 2 points for pearsonr
                mae_ld_pairs = mean_absolute_error(ld_real_pairs, ld_synth_pairs)
                rmse_ld_pairs = np.sqrt(mean_squared_error(ld_real_pairs, ld_synth_pairs))
                corr_ld_pairs, p_val_ld_pairs = pearsonr(ld_real_pairs, ld_synth_pairs)
                metrics_results[f"{name}_LD_Pairwise_MAE_ws{actual_ld_window}"] = mae_ld_pairs
                metrics_results[f"{name}_LD_Pairwise_RMSE_ws{actual_ld_window}"] = rmse_ld_pairs
                metrics_results[f"{name}_LD_Pairwise_Corr_ws{actual_ld_window}"] = corr_ld_pairs
                print(f"   {name} Pairwise LD (r^2) vs Real (Win: {actual_ld_window}): MAE={mae_ld_pairs:.4f}, RMSE={rmse_ld_pairs:.4f}, Corr={corr_ld_pairs:.4f}")
            else:
                print(f"   Not enough LD pairs for metrics for {name} (Window: {actual_ld_window})")
                metrics_results[f"{name}_LD_Pairwise_MAE_ws{actual_ld_window}"] = np.nan
                metrics_results[f"{name}_LD_Pairwise_RMSE_ws{actual_ld_window}"] = np.nan
                metrics_results[f"{name}_LD_Pairwise_Corr_ws{actual_ld_window}"] = np.nan


        fig_scatter_ld.suptitle(f"Pairwise LD (r²) Comparison (Window: {actual_ld_window} SNPs)", fontsize=16)
        fig_scatter_ld.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"ld_pairwise_scatter_combined_ws{actual_ld_window}.png")
        plt.close(fig_scatter_ld)
    else:
        print("   Skipping LD pairwise scatter plots as Real LD matrix could not be computed.")
        
    return metrics_results


def plot_combined_marginal_genotype_distribution(dosages_real_full, synth_datasets_info_list, output_dir):
    print("\n--- Evaluating Combined Aggregated Marginal Genotype Distribution Differences ---")
    metrics_results = {}
    num_gan = len(synth_datasets_info_list)
    genotype_labels = ['0/0 (Hom. Ref)', '0/1 (Het)', '1/1 (Hom. Alt)']
    
    # Create a figure for each genotype class, with subplots for each GAN
    # Or, one figure with 3 rows (genotypes) and num_gan columns. Let's try the latter.
    fig, axes = plt.subplots(3, num_gan, figsize=(5 * num_gan, 12), sharey='row', sharex='col') # Share y per row, x per col
    if num_gan == 1: # Adjust axes indexing if only one GAN
        axes = axes.reshape(3,1)


    # Collect all differences for global x-axis scaling per genotype class
    all_diffs_by_genotype = [[] for _ in range(3)] # List of lists for [diff_00, diff_01, diff_11]

    calculated_data = [] # To store calculated diffs for second pass plotting

    for synth_info in synth_datasets_info_list:
        name = synth_info['name']
        dosages_synth = synth_info['geno']
        num_snps_synth = synth_info['num_snps']
        
        dosages_real_sliced = dosages_real_full[:, :num_snps_synth]
        
        # Ensure synth data also uses its defined num_snps, although it should already be that size
        if dosages_synth.shape[1] != num_snps_synth:
            print(f"Warning: {name} synthetic data has {dosages_synth.shape[1]} SNPs, but info says {num_snps_synth}. Using actual: {dosages_synth.shape[1]}")
            # This shouldn't happen if data loading is correct based on SYNTH_DATA_PATHS
            # For safety, we could re-slice: dosages_synth = dosages_synth[:, :num_snps_synth]
        
        current_n_snps = min(dosages_real_sliced.shape[1], dosages_synth.shape[1]) # Should be num_snps_synth

        diff_prop_geno = [[] for _ in range(3)] # For current GAN: [00, 01, 11]

        for snp_idx in range(current_n_snps):
            prop_real_snp = np.zeros(3); prop_synth_snp = np.zeros(3)
            
            if dosages_real_sliced.shape[0] > 0:
                real_dosages_snp, real_counts_snp = np.unique(dosages_real_sliced[:, snp_idx], return_counts=True)
                for d_val, count in zip(real_dosages_snp, real_counts_snp):
                    if 0 <= d_val <= 2: prop_real_snp[d_val] = count / dosages_real_sliced.shape[0]
            
            if dosages_synth.shape[0] > 0:
                synth_dosages_snp, synth_counts_snp = np.unique(dosages_synth[:, snp_idx], return_counts=True)
                for d_val, count in zip(synth_dosages_snp, synth_counts_snp):
                    if 0 <= d_val <= 2: prop_synth_snp[d_val] = count / dosages_synth.shape[0]
            
            for geno_idx in range(3):
                diff = prop_synth_snp[geno_idx] - prop_real_snp[geno_idx]
                diff_prop_geno[geno_idx].append(diff)
                all_diffs_by_genotype[geno_idx].append(diff) # For global scaling

        calculated_data.append({'name': name, 'diffs': [np.array(d) for d in diff_prop_geno]})
        
        avg_abs_diff_overall_gan = (np.mean(np.abs(diff_prop_geno[0])) + 
                                    np.mean(np.abs(diff_prop_geno[1])) + 
                                    np.mean(np.abs(diff_prop_geno[2]))) / 3.0
        metrics_results[f"{name}_MarginalGeno_AvgAbsDiff_Overall"] = avg_abs_diff_overall_gan
        print(f"{name} Avg Abs Diff in Genotype Proportions: {avg_abs_diff_overall_gan:.4f}")
        for i_geno in range(3):
             metrics_results[f"{name}_MarginalGeno_MeanDiff_{genotype_labels[i_geno][:3]}"] = np.mean(diff_prop_geno[i_geno])


    # Determine common x-axis limits for each genotype row
    common_xlims = []
    for i_geno in range(3):
        if all_diffs_by_genotype[i_geno]:
            min_val = np.min(all_diffs_by_genotype[i_geno])
            max_val = np.max(all_diffs_by_genotype[i_geno])
            padding = (max_val - min_val) * 0.05
            common_xlims.append((min_val - padding, max_val + padding))
        else:
            common_xlims.append((-0.5, 0.5)) # Default if no data

    # Second pass: plot with consistent scales
    for i_gan, gan_data in enumerate(calculated_data):
        name = gan_data['name']
        diff_arrays = gan_data['diffs'] # This is [diff_00_gan, diff_01_gan, diff_11_gan]
        
        for i_geno, diff_data_geno in enumerate(diff_arrays): # Iterate through 00, 01, 11 for this GAN
            current_ax = axes[i_geno, i_gan]
            if diff_data_geno.size > 0:
                sns.histplot(diff_data_geno, kde=True, ax=current_ax, bins=30, stat="density") # Fewer bins for clarity
                current_ax.axvline(0, color='r', linestyle='--')
                current_ax.grid(True)
                current_ax.set_xlim(common_xlims[i_geno]) # Apply common xlim for this genotype row
            else:
                current_ax.text(0.5,0.5, "No Data", ha="center", va="center")

            if i_gan == 0: # First column GAN
                current_ax.set_ylabel(f'Density\nGenotype {genotype_labels[i_geno]}')
            if i_geno == 0: # First row (genotype 00)
                current_ax.set_title(f'{name}')
            if i_geno == 2: # Last row (genotype 11)
                 current_ax.set_xlabel('Difference (Synth Prop - Real Prop)')


    fig.suptitle('Distribution of Differences in Marginal Genotype Proportions (All SNPs)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "marginal_geno_diff_distributions_combined.png")
    plt.close(fig)
    return metrics_results


def plot_combined_pca_comparison(dosages_real_full, labels_real, synth_datasets_info_list, output_dir):
    print("\n--- Evaluating Combined PCA Comparison (Colored by Condition) ---")
    # PCA will be fit on a common subset of SNPs (first 7160 of real data)
    # and then applied to all datasets (using their first 7160 SNPs).
    num_gan = len(synth_datasets_info_list)
    
    # Determine common number of SNPs for PCA (min of real and GAN2)
    num_snps_for_pca = min(dosages_real_full.shape[1], SYNTH_DATA_PATHS["GAN_Paper1"]["num_snps"])
    print(f"   Using first {num_snps_for_pca} SNPs for PCA fitting and transformation across all datasets.")

    fig, axes = plt.subplots(1, num_gan + 1, figsize=(7 * (num_gan + 1), 6.5), sharex=True, sharey=True)
    if num_gan == 0: # Only real data
        axes = [axes] # Make iterable
    
    datasets_for_pca = []

    # Prepare Real Data for PCA
    n_real_samples = dosages_real_full.shape[0]
    pca_sample_size_real = min(PCA_SAMPLE_SIZE_EVAL, n_real_samples)
    real_idx = np.random.choice(n_real_samples, pca_sample_size_real, replace=False)
    
    pca_data_real_subset = dosages_real_full[real_idx, :num_snps_for_pca]
    pca_labels_real_subset = labels_real[real_idx]
    datasets_for_pca.append({'name': 'Real', 'data': pca_data_real_subset, 'labels': pca_labels_real_subset})

    # Prepare Synthetic Data for PCA
    for synth_info in synth_datasets_info_list:
        name = synth_info['name']
        dosages_synth = synth_info['geno']
        labels_synth = synth_info['pheno']
        
        n_synth_samples = dosages_synth.shape[0]
        pca_sample_size_synth = min(PCA_SAMPLE_SIZE_EVAL, n_synth_samples)
        synth_idx = np.random.choice(n_synth_samples, pca_sample_size_synth, replace=False)
        
        # Use the common num_snps_for_pca for synthetic data too
        pca_data_synth_subset = dosages_synth[synth_idx, :num_snps_for_pca]
        pca_labels_synth_subset = labels_synth[synth_idx]
        datasets_for_pca.append({'name': name, 'data': pca_data_synth_subset, 'labels': pca_labels_synth_subset})

    try:
        # Scale data: Fit on Real data subset, transform all
        scaler = StandardScaler()
        # Fit scaler ONLY on the real data used for PCA
        scaled_real_pca_data = scaler.fit_transform(datasets_for_pca[0]['data']) 
        
        transformed_pca_values = [{'name': 'Real', 'pca': scaled_real_pca_data, 'labels': datasets_for_pca[0]['labels']}]

        # Transform synthetic data subsets using the SAME scaler
        for i in range(1, len(datasets_for_pca)): # Start from 1 (synthetic datasets)
            scaled_synth_pca_data = scaler.transform(datasets_for_pca[i]['data'])
            transformed_pca_values.append({'name': datasets_for_pca[i]['name'], 
                                           'pca': scaled_synth_pca_data, 
                                           'labels': datasets_for_pca[i]['labels']})
        
        # PCA: Fit on scaled Real data, transform all scaled data
        pca = PCA(n_components=PCA_N_COMPONENTS_EVAL)
        # Fit PCA ONLY on the scaled real data
        pca_transformed_real = pca.fit_transform(transformed_pca_values[0]['pca'])
        print(f"   PCA Explained variance ratio (from Real data, {num_snps_for_pca} SNPs): {pca.explained_variance_ratio_}")

        plot_pca_data = [{'name': 'Real', 'coords': pca_transformed_real, 'labels': transformed_pca_values[0]['labels']}]

        # Transform other datasets using the SAME PCA model
        for i in range(1, len(transformed_pca_values)):
            pca_transformed_synth = pca.transform(transformed_pca_values[i]['pca'])
            plot_pca_data.append({'name': transformed_pca_values[i]['name'],
                                  'coords': pca_transformed_synth,
                                  'labels': transformed_pca_values[i]['labels']})

        # Determine common axis limits for PCA plots AFTER all transformations
        all_pc1 = np.concatenate([d['coords'][:, 0] for d in plot_pca_data if d['coords'].shape[0] > 0])
        all_pc2 = np.concatenate([d['coords'][:, 1] for d in plot_pca_data if d['coords'].shape[0] > 0])
        
        if all_pc1.size > 0 and all_pc2.size > 0:
            pc1_min, pc1_max = np.min(all_pc1), np.max(all_pc1)
            pc2_min, pc2_max = np.min(all_pc2), np.max(all_pc2)
            pc1_pad = (pc1_max - pc1_min) * 0.05
            pc2_pad = (pc2_max - pc2_min) * 0.05
            common_xlim_pca = (pc1_min - pc1_pad, pc1_max + pc1_pad)
            common_ylim_pca = (pc2_min - pc2_pad, pc2_max + pc2_pad)
        else: # Default if no data
            common_xlim_pca = (-3, 3)
            common_ylim_pca = (-3, 3)


        # Plotting
        for i, data_to_plot in enumerate(plot_pca_data):
            name = data_to_plot['name']
            coords = data_to_plot['coords']
            plot_labels = data_to_plot['labels']
            ax = axes[i]

            if coords.shape[0] > 0:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=plot_labels, cmap='viridis', alpha=0.6, s=15)
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)
            else:
                ax.text(0.5,0.5, "No Data", ha="center", va="center")

            ax.set_title(f'PCA of {name} Data\n({num_snps_for_pca} SNPs, Colored by Condition)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})' if i == 0 else '')
            ax.grid(True)
            ax.set_xlim(common_xlim_pca)
            ax.set_ylim(common_ylim_pca)

        fig.suptitle(f'PCA Comparison (Scaler & PCA fit on Real, {num_snps_for_pca} SNPs, Applied to All)', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / f"pca_comparison_combined_common_snps{num_snps_for_pca}.png")
        print(f"   Combined PCA plot (using {num_snps_for_pca} SNPs) saved.")
    except Exception as e:
        print(f"   Error during Combined PCA analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)
    return {} # PCA plots are visual, specific metrics can be added if needed (e.g. MMD on PC distributions)

def evaluate_all_real_vs_fake_classifiers(dosages_real_full, synth_datasets_info_list, output_dir):
    print("\n--- Evaluating Real-vs-Fake Classifiers (one for each GAN) ---")
    metrics_results = {}
    results_text = [] # For a small summary plot
    num_gan = len(synth_datasets_info_list)

    for synth_info in synth_datasets_info_list:
        name = synth_info['name']
        dosages_synth = synth_info['geno']
        num_snps_synth = synth_info['num_snps']

        # Slice real data to match current synthetic data's SNP count
        dosages_real_sliced = dosages_real_full[:, :num_snps_synth]
        
        # Ensure synth data also has the correct number of SNPs (it should by design)
        # dosages_synth_sliced = dosages_synth[:, :num_snps_synth]

        n_real_effective = dosages_real_sliced.shape[0]
        n_synth_effective = dosages_synth.shape[0]
        
        n_min = min(n_real_effective, n_synth_effective)

        if n_min < 20: # Need enough samples for train/test split
            print(f"Not enough samples ({n_min}) for Real-vs-Fake classifier for {name}. Skipping.")
            metrics_results[f"{name}_RealVsFake_AUC"] = np.nan
            results_text.append(f"{name}: N/A (low samples)")
            continue

        idx_real = np.random.choice(n_real_effective, n_min, replace=False)
        idx_synth = np.random.choice(n_synth_effective, n_min, replace=False)

        X_rvf = np.vstack((dosages_real_sliced[idx_real], dosages_synth[idx_synth]))
        y_rvf = np.array([0]*n_min + [1]*n_min) # 0 for Real, 1 for Fake (Synthetic)
        
        try:
            X_train_rvf, X_test_rvf, y_train_rvf, y_test_rvf = train_test_split(X_rvf, y_rvf, test_size=0.3, random_state=42, stratify=y_rvf)
        except ValueError: # If stratification fails (e.g. too few samples of one class in a small test set)
            X_train_rvf, X_test_rvf, y_train_rvf, y_test_rvf = train_test_split(X_rvf, y_rvf, test_size=0.3, random_state=42)

        if X_train_rvf.shape[0] < 10 or X_test_rvf.shape[0] < 10 : # Check if splits are too small
             print(f"Train/test split too small for Real-vs-Fake for {name}. Skipping.")
             metrics_results[f"{name}_RealVsFake_AUC"] = np.nan
             results_text.append(f"{name}: N/A (split error)")
             continue
        
        try:
            model_rvf = LogisticRegression(solver='liblinear', random_state=42, max_iter=300, C=0.1) # Added C and max_iter
            model_rvf.fit(X_train_rvf, y_train_rvf)
            y_pred_proba_rvf = model_rvf.predict_proba(X_test_rvf)[:, 1]
            auc_rvf = roc_auc_score(y_test_rvf, y_pred_proba_rvf)
            print(f"{name} Real-vs-Fake Classifier AUC: {auc_rvf:.4f} (SNPs: {num_snps_synth}, Samples/class: {n_min})")
            metrics_results[f"{name}_RealVsFake_AUC"] = auc_rvf
            results_text.append(f"{name} (SNPs: {num_snps_synth}): {auc_rvf:.4f}")
        except Exception as e:
            print(f"Error training/evaluating Real-vs-Fake for {name}: {e}")
            metrics_results[f"{name}_RealVsFake_AUC"] = np.nan
            results_text.append(f"{name}: Error")

    # Simple bar plot for AUCs
    if metrics_results:
        gan_names = [si['name'] for si in synth_datasets_info_list]
        aucs = [metrics_results.get(f"{name}_RealVsFake_AUC", 0.5) for name in gan_names] # Default to 0.5 if NaN for plotting
        
        fig, ax = plt.subplots(figsize=(max(8, num_gan * 2.5), 5))
        bars = ax.bar(gan_names, aucs, color=['skyblue', 'salmon', 'lightgreen'][:num_gan])
        ax.set_ylabel('AUC (Real=0, Fake=1)')
        ax.set_title('Real-vs-Fake Classifier AUC for each GAN')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='grey', linestyle='--', label='Ideal (Indistinguishable)')
        ax.legend()
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "real_vs_fake_auc_summary.png")
        plt.close(fig)

    return metrics_results

# --- Main Execution ---
if __name__ == '__main__':
    start_time = time.time()
    all_eval_metrics_combined = {}

    # 1. Load Real Data
    print("--- Loading Real Data ---")
    dosages_real = load_dosage_data(REAL_GENO_PATH, expected_snps=ORIGINAL_GENOTYPE_LEN, label="Real")
    labels_real = load_phenotype_data(REAL_PHENO_PATH, label="Real")
    if dosages_real.shape[0] != labels_real.shape[0]:
        raise ValueError("Mismatch in number of samples between real genotype and phenotype files.")
    print(f"Real data shapes: Dosages {dosages_real.shape}, Labels {labels_real.shape}")

    # 2. Load Synthetic Data
    print("\n--- Loading Synthetic Data ---")
    synthetic_datasets = []
    for name, paths in SYNTH_DATA_PATHS.items():
        print(f"Processing {name}...")
        try:
            dosages_synth = load_dosage_data(paths["geno"], expected_snps=paths["num_snps"], label=name)
            labels_synth = load_phenotype_data(paths["pheno"], label=name)
            if dosages_synth.shape[0] != labels_synth.shape[0]:
                print(f"Warning: Mismatch samples for {name}. Geno: {dosages_synth.shape[0]}, Pheno: {labels_synth.shape[0]}. Skipping this dataset.")
                continue
            if dosages_synth.shape[1] != paths["num_snps"]:
                 print(f"Warning: {name} has {dosages_synth.shape[1]} SNPs, config expected {paths['num_snps']}. Using actual, but check config.")
                 # Update num_snps in paths if it's just a config error and we want to proceed
                 # paths['num_snps'] = dosages_synth.shape[1] # Be careful with this auto-correction

            synthetic_datasets.append({
                "name": name,
                "geno": dosages_synth,
                "pheno": labels_synth,
                "num_snps": dosages_synth.shape[1] # Use actual loaded SNP count
            })
            print(f"Loaded {name}: Dosages {dosages_synth.shape}, Labels {labels_synth.shape}")
        except FileNotFoundError as e:
            print(f"Skipping {name} due to missing file: {e}")
        except ValueError as e:
            print(f"Skipping {name} due to value error during loading: {e}")
        except Exception as e:
            print(f"Skipping {name} due to an unexpected error: {e}")


    if not synthetic_datasets:
        print("No synthetic datasets loaded. Exiting evaluation.")
        exit()
    
    print(f"\nSuccessfully loaded {len(synthetic_datasets)} synthetic dataset(s) for evaluation.")

    # 3. Run Combined Evaluations
    # Note: Pass dosages_real (full) and labels_real to functions.
    # The functions will handle slicing of real data internally when comparing with GANs having fewer SNPs.
    
    metrics_af = plot_combined_allele_frequencies(dosages_real, labels_real, synthetic_datasets, OUTPUT_DIR)
    all_eval_metrics_combined.update(metrics_af)
    gc.collect()

    metrics_chi2 = plot_combined_chi_squared_distribution(dosages_real, labels_real, synthetic_datasets, OUTPUT_DIR)
    all_eval_metrics_combined.update(metrics_chi2)
    gc.collect()
    
    # LD analysis needs dosages only. Labels are not directly used in LD calculation itself.
    metrics_ld = plot_combined_ld_structure(dosages_real, synthetic_datasets, OUTPUT_DIR, ld_window_config=LD_WINDOW_FOR_PLOTS)
    all_eval_metrics_combined.update(metrics_ld)
    gc.collect()

    metrics_mgd = plot_combined_marginal_genotype_distribution(dosages_real, synthetic_datasets, OUTPUT_DIR)
    all_eval_metrics_combined.update(metrics_mgd)
    gc.collect()

    # PCA comparison uses dosages and labels (for coloring points)
    plot_combined_pca_comparison(dosages_real, labels_real, synthetic_datasets, OUTPUT_DIR)
    # PCA is mostly visual, specific metrics could be added if needed (e.g. MMD on PC distributions)
    gc.collect()

    # Real-vs-Fake classifiers (one per GAN)
    metrics_rvf = evaluate_all_real_vs_fake_classifiers(dosages_real, synthetic_datasets, OUTPUT_DIR)
    all_eval_metrics_combined.update(metrics_rvf)
    gc.collect()

    # Add other evaluations if desired, e.g., discriminator scores, conditional generation quality
    # These would also need to be adapted to loop through synthetic_datasets and handle SNP differences.
    # For example, evaluate_discriminator_scores would need a discriminator model compatible with varying SNP inputs
    # or you'd need separate discriminators, or only evaluate on common SNPs.

    # 4. Save All Metrics
    metrics_file = OUTPUT_DIR / "all_evaluation_metrics_combined.txt"
    with open(metrics_file, 'w') as f:
        f.write("Combined GAN Evaluation Metrics:\n==============================\n")
        # Sort metrics by key for consistent output
        for key in sorted(all_eval_metrics_combined.keys()):
            value = all_eval_metrics_combined[key]
            if isinstance(value, float):
                f.write(f"{key}: {value:.4g}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\nAll quantitative metrics saved to: {metrics_file}")
    print(f"Plots and metrics saved in: {OUTPUT_DIR.resolve()}")
    total_time = time.time() - start_time
    print(f"\n--- Combined Evaluation Script Finished in {total_time:.2f} seconds ---")
