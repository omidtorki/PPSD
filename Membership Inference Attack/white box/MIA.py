import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Configuration ---
CHECKPOINT_DIR = '.training_checkpoints/UserSNP_ACGAN_CondAF_MMDChiSqLoss'
CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30] 

MEMBER_GENO_PATH = '../data_split/member_geno.txt'
MEMBER_PHENO_PATH = '../data_split/member_pheno.txt'
NON_MEMBER_GENO_PATH = '../data_split/non_member_geno.txt'
NON_MEMBER_PHENO_PATH = '../data_split/non_member_pheno.txt'

GENOTYPE_LEN = 10000
CHANNELS = 3
NUM_CLASSES = 2
TEST_SAMPLE_COUNT = 100 
RANDOM_STATE = 42 # Use a fixed random state for reproducibility

# --- Data Loading (Identical) ---
def load_and_prepare_test_data(geno_path, label_path):
    labels = np.loadtxt(label_path, dtype=int)
    dataset = np.loadtxt(geno_path, dtype=np.float32)
    dataset_int = dataset.astype(np.int32)
    X_one_hot = to_categorical(dataset_int, num_classes=CHANNELS)
    labels_reshaped = labels.reshape(-1, 1)
    return X_one_hot.astype(np.float32), labels_reshaped.astype(np.int32)

# --- Main White-Box Attack Logic ---
def run_white_box_rank_attack_on_checkpoints():
    print("--- Starting White-Box Membership Inference Attack on Checkpoints (Rank-Based) ---")
    
    # 1. Load all available member and non-member data
    print("Loading all member and non-member data...")
    member_X_all, member_y_all = load_and_prepare_test_data(MEMBER_GENO_PATH, MEMBER_PHENO_PATH)
    non_member_X_all, non_member_y_all = load_and_prepare_test_data(NON_MEMBER_GENO_PATH, NON_MEMBER_PHENO_PATH)
    
    # --- *** NEW: Randomly select a fixed subset for testing *** ---
    print(f"Randomly selecting {TEST_SAMPLE_COUNT} members and {TEST_SAMPLE_COUNT} non-members for the fixed test set...")
    
    # Set the random seed to ensure the *same* random samples are chosen every time
    np.random.seed(RANDOM_STATE)

    # Randomly choose indices without replacement
    member_indices = np.random.choice(len(member_X_all), TEST_SAMPLE_COUNT, replace=False)
    non_member_indices = np.random.choice(len(non_member_X_all), TEST_SAMPLE_COUNT, replace=False)

    # Create the fixed test set using the random indices
    member_X_test = member_X_all[member_indices]
    member_y_test = member_y_all[member_indices]
    non_member_X_test = non_member_X_all[non_member_indices]
    non_member_y_test = non_member_y_all[non_member_indices]
    
    # Clear the full datasets from memory to save RAM
    del member_X_all, member_y_all, non_member_X_all, non_member_y_all

    # --- The rest of the script is the same ---
    
    test_X = np.concatenate([member_X_test, non_member_X_test], axis=0)
    test_y_for_model = np.concatenate([member_y_test, non_member_y_test], axis=0)
    
    attack_true_labels = np.concatenate([np.ones(len(member_X_test)), np.zeros(len(non_member_X_test))])
    
    n_members_in_test = len(member_X_test)
    print(f"Fixed random test set created with {n_members_in_test} members and {len(non_member_X_test)} non-members.")

    # 2. Iterate through each checkpoint
    results = {'epochs': [], 'accuracy': [], 'tps': [], 'tns': [], 'fps': [], 'fns': []}

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n--- Evaluating Checkpoint for Epoch {epoch} ---")
        model_path = os.path.join(CHECKPOINT_DIR, f'discriminator_epoch_{epoch}.keras')
        
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint file not found, skipping epoch {epoch}. Searched at: {model_path}")
            continue

        target_discriminator = load_model(model_path)
        
        pred_outputs = target_discriminator.predict([test_X, test_y_for_model], verbose=0)
        adversarial_scores = pred_outputs[0].flatten()

        sorted_indices = np.argsort(adversarial_scores)[::-1]
        attack_predicted_labels = np.zeros_like(attack_true_labels)
        top_n_indices = sorted_indices[:n_members_in_test]
        attack_predicted_labels[top_n_indices] = 1
        
        accuracy = accuracy_score(attack_true_labels, attack_predicted_labels)
        tn, fp, fn, tp = confusion_matrix(attack_true_labels, attack_predicted_labels, labels=[0, 1]).ravel()
        
        results['epochs'].append(epoch)
        results['accuracy'].append(accuracy)
        results['tps'].append(tp); results['tns'].append(tn); results['fps'].append(fp); results['fns'].append(fn)

        print(f"Epoch {epoch} Rank-Attack Results: Accuracy={accuracy:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        del target_discriminator
        tf.keras.backend.clear_session()

    # 3. Save and Plot the final results (code is identical)
    if not results['epochs']:
        print("\nNo valid checkpoints were found or evaluated. Exiting.")
        return

    results_filename = "whitebox_checkpoints_rank_results.csv"
    results_array = np.array([
        results['epochs'], results['accuracy'], 
        results['tps'], results['tns'], results['fps'], results['fns']
    ]).T
    np.savetxt(results_filename, results_array, delimiter=',', header='Epoch,Accuracy,TP,TN,FP,FN', comments='')
    print(f"\nDetailed results saved to {results_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(results['epochs'], results['accuracy'], marker='o', linestyle='-')
    plt.title('White-Box Attack Vulnerability vs. Training Epoch (Rank-Based)')
    plt.xlabel('Training Epoch of Target Model')
    plt.ylabel('Membership Inference Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (50%)')
    plt.xticks(CHECKPOINT_EPOCHS)
    plt.grid(True)
    plt.legend()
    plt.ylim(0.4, 1.05)
    
    plot_filename = "whitebox_rank_vulnerability_over_epochs.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    run_white_box_rank_attack_on_checkpoints()
