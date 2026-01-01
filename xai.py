import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # For creating tables

# --- Constants and Paths ---
GENOTYPE_LEN = 10000
CHANNELS = 3
NUM_CLASSES = 2
NOISE_DIM = 100
SAVED_GENERATOR_PATH = '../generator_epoch_30.keras'
SAVED_DISCRIMINATOR_PATH = '../discriminator_epoch_30.keras'
USER_GENO_PATH = "../../../geno_10000_2.txt" 
USER_LABEL_PATH = "../../../pheno.txt"



def load_user_data_for_xai(geno_path, label_path, num_samples_to_load=100):
    if not os.path.exists(label_path): raise FileNotFoundError(f"Label file not found: {label_path}")
    all_labels = np.loadtxt(label_path, dtype=int)
    if not os.path.exists(geno_path): raise FileNotFoundError(f"Genotype file not found: {geno_path}")

    print(f"Loading first {num_samples_to_load} samples from {geno_path} for XAI...")
    try:
        all_dataset_dosages = np.loadtxt(geno_path, dtype=np.float32, max_rows=num_samples_to_load)
    except Exception as e:
        print(f"Could not load genotype data with np.loadtxt directly. Error: {e}")
        raise

    if all_dataset_dosages.ndim == 1 and num_samples_to_load == 1 :
        all_dataset_dosages = np.expand_dims(all_dataset_dosages, axis=0)
    
    current_samples_loaded = all_dataset_dosages.shape[0]
    if current_samples_loaded < num_samples_to_load:
        print(f"Warning: Requested {num_samples_to_load} samples, but only {current_samples_loaded} were loaded from genotype file.")
        num_samples_to_load = current_samples_loaded

    if all_dataset_dosages.shape[1] != GENOTYPE_LEN:
        print(f"Warning: Genotype file has {all_dataset_dosages.shape[1]} SNPs, but GENOTYPE_LEN is {GENOTYPE_LEN}. Adjusting by truncating/padding.")
        # Simple truncation, more sophisticated padding might be needed if file has fewer SNPs
        if all_dataset_dosages.shape[1] > GENOTYPE_LEN:
            all_dataset_dosages = all_dataset_dosages[:, :GENOTYPE_LEN]
        else: # Pad with zeros if fewer SNPs
            padding = np.zeros((num_samples_to_load, GENOTYPE_LEN - all_dataset_dosages.shape[1]))
            all_dataset_dosages = np.concatenate([all_dataset_dosages, padding], axis=1)


    nan_mask = np.isnan(all_dataset_dosages)
    if np.any(nan_mask):
        all_dataset_dosages[nan_mask] = 0.0

    dataset_int = all_dataset_dosages.astype(np.int32)
    min_val, max_val = np.min(dataset_int), np.max(dataset_int)
    if min_val < 0 or max_val > 2:
        raise ValueError(f"Input geno.txt must contain dosages 0, 1, or 2. Found range [{min_val}, {max_val}].")

    X_one_hot = to_categorical(dataset_int, num_classes=CHANNELS).astype(np.float32)
    
    if all_labels.shape[0] < num_samples_to_load:
        print(f"Warning: Requested {num_samples_to_load} labels, but only {all_labels.shape[0]} were available.")
        num_samples_to_load = min(all_labels.shape[0], X_one_hot.shape[0]) # Ensure consistency
        X_one_hot = X_one_hot[:num_samples_to_load, :, :]
        
    labels_subset = all_labels[:num_samples_to_load].astype(np.int32)

    print(f"Loaded data shapes for XAI: X={X_one_hot.shape}, Y={labels_subset.shape}")
    if X_one_hot.shape[0] == 0:
        raise ValueError("No data loaded. Check paths and file contents.")
    return X_one_hot, labels_subset

@tf.function
def get_gradient_input_attributions(discriminator_model, genotype_input_one_hot, label_input_for_model, target_output_index):
    genotype_input_tensor = tf.convert_to_tensor(genotype_input_one_hot, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(genotype_input_tensor)
        predictions = discriminator_model([genotype_input_tensor, label_input_for_model], training=False)
        target_output = predictions[target_output_index]
        scalar_target_output = target_output[0,0] # Assumes batch_size is 1 for per-sample attribution
    gradients = tape.gradient(scalar_target_output, genotype_input_tensor)
    if gradients is None:
        raise RuntimeError("Gradients are None. Check model connectivity and watched tensors.")
    attributions = gradients * genotype_input_tensor
    return attributions

def summarize_attributions(attributions_batch, top_n=5):
    if attributions_batch is None or attributions_batch.shape[0] == 0:
        print("No attributions to summarize.")
        return [], []
    attributions_per_snp_batch = tf.reduce_sum(attributions_batch, axis=-1)
    abs_attributions_per_snp_batch = tf.abs(attributions_per_snp_batch)
    avg_abs_attributions_per_snp = tf.reduce_mean(abs_attributions_per_snp_batch, axis=0)
    actual_top_n = min(top_n, avg_abs_attributions_per_snp.shape[0])
    if actual_top_n == 0 : return [], []
    top_indices = tf.argsort(avg_abs_attributions_per_snp, direction='DESCENDING')[:actual_top_n].numpy()
    top_scores = tf.gather(avg_abs_attributions_per_snp, top_indices).numpy()
    print(f"Top {actual_top_n} most influential SNPs (avg abs attribution scores):")
    for i in range(actual_top_n):
        print(f"  SNP Index: {top_indices[i]}, Score: {top_scores[i]:.4f}")
    return top_indices, top_scores


def plot_top_n_snps(avg_abs_attributions_per_snp, top_n=50, title="Top N Influential SNPs", filename_suffix=""):
    if avg_abs_attributions_per_snp is None or avg_abs_attributions_per_snp.shape[0] == 0:
        print(f"No data to plot for {title}")
        return
    actual_top_n = min(top_n, avg_abs_attributions_per_snp.shape[0])
    if actual_top_n == 0 : return
    top_indices = tf.argsort(avg_abs_attributions_per_snp, direction='DESCENDING')[:actual_top_n].numpy()
    top_scores = tf.gather(avg_abs_attributions_per_snp, top_indices).numpy()
    plt.figure(figsize=(15, 7))
    sns.barplot(x=[str(i) for i in top_indices], y=top_scores, palette="viridis")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("SNP Index")
    plt.ylabel("Average Absolute Attribution Score")
    plt.title(title)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').replace('/', '_').lower()}{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved bar plot: {filename}")
    plt.close()

def plot_attribution_heatmap(attributions_batch, num_samples_to_plot=10, num_snps_to_plot=50, title="SNP Attribution Heatmap", filename_suffix=""):
    if attributions_batch is None or attributions_batch.shape[0] == 0:
        print(f"No attributions to plot heatmap for {title}")
        return
    attributions_per_snp_batch = tf.reduce_sum(tf.abs(attributions_batch), axis=-1).numpy()
    actual_num_samples = min(num_samples_to_plot, attributions_per_snp_batch.shape[0])
    actual_num_snps = min(num_snps_to_plot, attributions_per_snp_batch.shape[1])
    if actual_num_snps == 0 or actual_num_samples == 0 : return
    mean_attr_across_batch = np.mean(attributions_per_snp_batch, axis=0)
    top_snp_indices = np.argsort(mean_attr_across_batch)[::-1][:actual_num_snps]
    data_to_plot = attributions_per_snp_batch[:actual_num_samples, :][:, top_snp_indices] # Corrected slicing
    sample_labels = [f"Sample {i}" for i in range(actual_num_samples)]
    snp_labels = [f"SNP {i}" for i in top_snp_indices]
    plt.figure(figsize=(min(20, actual_num_snps * 0.4 + 2), min(8, actual_num_samples * 0.5 + 2)))
    sns.heatmap(data_to_plot, xticklabels=snp_labels, yticklabels=sample_labels, cmap="viridis", annot=False)
    plt.xlabel(f"Top {actual_num_snps} SNP Indices")
    plt.ylabel("Sample Index")
    plt.title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').replace('/', '_').lower()}{filename_suffix}_heatmap.png"
    plt.savefig(filename)
    print(f"Saved heatmap: {filename}")
    plt.close()


def get_top_k_percent_snps_indices(snp_importance_scores, percentage):

    if snp_importance_scores is None or snp_importance_scores.shape[0] == 0:
        return set()
    k = int(np.ceil(len(snp_importance_scores) * (percentage / 100.0)))
    k = min(k, len(snp_importance_scores)) # Ensure k is not more than available SNPs
    if k == 0: return set()
    top_indices = tf.argsort(snp_importance_scores, direction='DESCENDING')[:k].numpy()
    return set(top_indices)

def calculate_overlap_stats(avg_attr_scores_dict, percentages_to_check, total_snps):
    print("\n--- SNP Overlap Analysis ---")
    
    comparisons = [
        ("RealAdv_vs_FakeAdv", "Real-Adv", "Fake-Adv"),
        ("RealClf_vs_FakeClf", "Real-Clf", "Fake-Clf"),
        ("RealAdv_vs_RealClf", "Real-Adv", "Real-Clf"),
        ("FakeAdv_vs_FakeClf", "Fake-Adv", "Fake-Clf")
    ]
    
    overlap_data = []

    for p in percentages_to_check:
        row_data = {"Percentage": f"Top {p}%"}
        num_top_snps = int(np.ceil(total_snps * (p / 100.0)))
        if num_top_snps == 0: continue

        for comp_name, key1, key2 in comparisons:
            scores1 = avg_attr_scores_dict.get(key1)
            scores2 = avg_attr_scores_dict.get(key2)

            if scores1 is None or scores2 is None:
                row_data[comp_name] = "N/A"
                continue

            top_snps1 = get_top_k_percent_snps_indices(scores1, p)
            top_snps2 = get_top_k_percent_snps_indices(scores2, p)
            
            if not top_snps1: # If one of the sets is empty (e.g. due to k=0 from very small percentage or no SNPs)
                row_data[comp_name] = "0.00%"
                continue

            intersection_size = len(top_snps1.intersection(top_snps2))

            overlap_percent = (intersection_size / len(top_snps1)) * 100 if len(top_snps1) > 0 else 0
            row_data[comp_name] = f"{overlap_percent:.2f}%"
        overlap_data.append(row_data)

    if not overlap_data:
        print("No overlap data generated.")
        return

    df_overlap = pd.DataFrame(overlap_data)
    print(df_overlap.to_string(index=False))
    
    # Save to CSV
    df_overlap.to_csv("snp_overlap_analysis.csv", index=False)
    print("\nOverlap analysis saved to snp_overlap_analysis.csv")

# --- Modified plotting functions for separate box plots ---

def plot_adversarial_attribution_distributions(avg_attr_scores_dict, title="Distribution of SNP Attribution Scores - Adversarial Output"):
    """ Plots box plots for adversarial attribution scores only. """
    print(f"\n--- Plotting {title} (Box Plots) ---")
    
    plot_data = []
    labels = []
    
    # Only include 'Real-Adv' and 'Fake-Adv'
    if "Real-Adv" in avg_attr_scores_dict and avg_attr_scores_dict["Real-Adv"] is not None:
        plot_data.append(avg_attr_scores_dict["Real-Adv"].numpy())
        labels.append("Real-Adv")
    if "Fake-Adv" in avg_attr_scores_dict and avg_attr_scores_dict["Fake-Adv"] is not None:
        plot_data.append(avg_attr_scores_dict["Fake-Adv"].numpy())
        labels.append("Fake-Adv")
            
    if not plot_data:
        print(f"No data available for {title}.")
        return

    plt.figure(figsize=(8, 7)) # Adjust figure size for fewer plots
    sns.boxplot(data=plot_data, palette="Set1") # Using a different palette for clarity
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.ylabel("Average Absolute Attribution Score")
    plt.title(title)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').lower()}_boxplots.png"
    plt.savefig(filename)
    print(f"Saved box plot distributions: {filename}")
    plt.close()

def plot_classifier_attribution_distributions(avg_attr_scores_dict, title="Distribution of SNP Attribution Scores - Classifier Output"):
    """ Plots box plots for classifier attribution scores only. """
    print(f"\n--- Plotting {title} (Box Plots) ---")
    
    plot_data = []
    labels = []
    
    # Only include 'Real-Clf' and 'Fake-Clf'
    if "Real-Clf" in avg_attr_scores_dict and avg_attr_scores_dict["Real-Clf"] is not None:
        plot_data.append(avg_attr_scores_dict["Real-Clf"].numpy())
        labels.append("Real-Clf")
    if "Fake-Clf" in avg_attr_scores_dict and avg_attr_scores_dict["Fake-Clf"] is not None:
        plot_data.append(avg_attr_scores_dict["Fake-Clf"].numpy())
        labels.append("Fake-Clf")
            
    if not plot_data:
        print(f"No data available for {title}.")
        return

    plt.figure(figsize=(8, 7)) # Adjust figure size for fewer plots
    sns.boxplot(data=plot_data, palette="Set2") # Using a different palette for clarity
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.ylabel("Average Absolute Attribution Score")
    plt.title(title)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').lower()}_boxplots.png"
    plt.savefig(filename)
    print(f"Saved box plot distributions: {filename}")
    plt.close()



if __name__ == '__main__':
    # --- GPU Setup ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s).")
        except RuntimeError as e: print(e)
    else: print("No GPU detected, running on CPU.")

    # --- Load Models ---
    print("\nLoading trained models...")
    if not os.path.exists(SAVED_DISCRIMINATOR_PATH) or not os.path.exists(SAVED_GENERATOR_PATH):
        print(f"ERROR: Model files not found. D: {SAVED_DISCRIMINATOR_PATH}, G: {SAVED_GENERATOR_PATH}")
        exit()
    try:
        discriminator = load_model(SAVED_DISCRIMINATOR_PATH, compile=False)
        generator = load_model(SAVED_GENERATOR_PATH, compile=False)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}"); exit()

    # --- Load real data ---
    print("\nLoading real samples for XAI...")
    num_xai_samples = 100
    try:
        X_real_sample, Y_real_sample = load_user_data_for_xai(USER_GENO_PATH, USER_LABEL_PATH, num_samples_to_load=num_xai_samples)
        num_xai_samples = X_real_sample.shape[0] # Update based on actual loaded samples
        if num_xai_samples == 0: print("No real samples loaded, exiting."); exit()
    except Exception as e:
        print(f"Error loading real data for XAI: {e}"); exit()

    # --- Generate fake data ---
    print("\nGenerating fake samples for XAI...")
    noise_sample = tf.random.normal([num_xai_samples, NOISE_DIM])
    fake_target_labels_sample = np.random.randint(0, NUM_CLASSES, size=(num_xai_samples,)).astype(np.int32)
    X_fake_sample_probs = generator([noise_sample, tf.expand_dims(fake_target_labels_sample, axis=-1)], training=False)

    # --- Perform XAI Analysis and Store Average Attributions ---
    analysis_scenarios_map = {
        "Real-Adv": {"data_x": X_real_sample, "data_y": Y_real_sample, "target_idx": 0, "title": "REAL Samples - Adversarial"},
        "Fake-Adv": {"data_x": X_fake_sample_probs, "data_y": fake_target_labels_sample, "target_idx": 0, "title": "FAKE Samples - Adversarial"},
        "Real-Clf": {"data_x": X_real_sample, "data_y": Y_real_sample, "target_idx": 1, "title": "REAL Samples - Classifier"},
        "Fake-Clf": {"data_x": X_fake_sample_probs, "data_y": fake_target_labels_sample, "target_idx": 1, "title": "FAKE Samples - Classifier"}
    }
    
    # Dictionary to store average absolute attribution scores per SNP for each scenario
    avg_attr_scores_all_scenarios = {}

    for scenario_key, params in analysis_scenarios_map.items():
        print(f"\n--- XAI for {params['title']} ---")
        current_attributions_list = []
        data_x, data_y, target_idx = params["data_x"], params["data_y"], params["target_idx"]

        if data_x.shape[0] == 0:
            print(f"Skipping {params['title']} as no data is available.")
            avg_attr_scores_all_scenarios[scenario_key] = None
            continue

        for i in range(data_x.shape[0]):
            single_x_sample = tf.expand_dims(data_x[i], axis=0)
            single_y_label_for_model = tf.constant([[data_y[i]]], dtype=tf.int32)
            attrs = get_gradient_input_attributions(discriminator, single_x_sample, single_y_label_for_model, target_idx)
            current_attributions_list.append(attrs)

        if not current_attributions_list:
            print(f"No attributions generated for {params['title']}.")
            avg_attr_scores_all_scenarios[scenario_key] = None
            continue
            
        aggregated_attrs_current_scenario = tf.concat(current_attributions_list, axis=0)
        summarize_attributions(aggregated_attrs_current_scenario, top_n=5)
        
        # Calculate and store average absolute attributions per SNP
        avg_abs_attrs_per_snp = tf.reduce_mean(tf.abs(tf.reduce_sum(aggregated_attrs_current_scenario, axis=-1)), axis=0)
        avg_attr_scores_all_scenarios[scenario_key] = avg_abs_attrs_per_snp
        
        # Standard Visualizations
        plot_top_n_snps(avg_abs_attrs_per_snp, top_n=min(50, GENOTYPE_LEN), title=params["title"], filename_suffix=f"_{scenario_key.lower()}")
        plot_attribution_heatmap(aggregated_attrs_current_scenario,
                                 num_samples_to_plot=min(10, num_xai_samples),
                                 num_snps_to_plot=min(50, GENOTYPE_LEN),
                                 title=params["title"],
                                 filename_suffix=f"_{scenario_key.lower()}")
        gc.collect()

    # --- Advanced Statistical Analysis & New Separate Plots ---
    percentages_for_overlap = [1, 5, 10, 25, 50]
    calculate_overlap_stats(avg_attr_scores_all_scenarios, percentages_for_overlap, GENOTYPE_LEN)
    
    # Call the new separate plotting functions
    plot_adversarial_attribution_distributions(avg_attr_scores_all_scenarios)
    plot_classifier_attribution_distributions(avg_attr_scores_all_scenarios)

    print("\n--- XAI Script with Statistical Analysis Finished ---")
    print("Plots and overlap CSV saved in the current directory.")

    # Clean up
    del generator, discriminator, X_real_sample, Y_real_sample, X_fake_sample_probs, avg_attr_scores_all_scenarios
    gc.collect()
