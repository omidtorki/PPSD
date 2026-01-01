import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# No extra custom objects seem to be defined explicitly for layers in this generator,
# but Keras can sometimes save optimizer state that might implicitly link to custom functions
# if they were used in losses. For inference, compile=False is often best.
from pathlib import Path

# --- Global variables for data synthesis ---
Num_Contrl = 3959  # Number of control samples to synthesize (label 0)
Num_Case = 4922    # Number of case samples to synthesize (label 1)

# --- Configuration: MUST MATCH ACGAN_CondAF_MMDChiSq TRAINING SETUP AND SAVED MODEL ---
# Model saving details from training script:
# Default final saved name:
SAVED_GENERATOR_FILENAME = ".training_checkpoints/UserSNP_ACGAN_CondAF_MMDChiSqLoss/generator_epoch_30.keras"
# Or if you want to use a model from a specific epoch:
# CHECKPOINT_DIR_FOR_LOADING = ".training_checkpoints/UserSNP_ACGAN_CondAF_MMDChiSqLoss"
# EPOCH_TO_LOAD = 100 # Example: last epoch if EPOCHS=100 and SAVE_INTERVAL=10
# SAVED_GENERATOR_FILENAME = f"generator_epoch_{EPOCH_TO_LOAD}.keras"
# MODEL_BASE_PATH = CHECKPOINT_DIR_FOR_LOADING # If loading from checkpoint dir
MODEL_BASE_PATH = "." # Assumes the final .keras file is in the same directory as this script

# Parameters from ACGAN_CondAF_MMDChiSq training script:
NOISE_DIM = 100
GENOTYPE_LEN = 10000 # Number of SNPs
CHANNELS = 3         # One-hot encoding channels (0, 1, 2 -> [1,0,0], [0,1,0], [0,0,1])
NUM_CLASSES = 2      # Number of phenotype classes (0 for control, 1 for case)

# --- Output file names ---
GENO_OUTPUT_FILE_ACGAN_MMD = "geno_synth_ACGAN_MMD.txt"
PHENO_OUTPUT_FILE_ACGAN_MMD = "pheno_synth_ACGAN_MMD.txt"

# --- Custom Objects (Placeholder - likely not needed for this generator's structure for inference) ---
# If model loading fails due to unrecognized custom losses or metrics referenced
# by the optimizer state (even with compile=False), you might need to define dummy versions here.
# However, for inference of the generator alone, this is usually not an issue.
custom_objects = {}


def generate_synthetic_data_acgan_mmd():
    """
    Loads a pre-trained ACGAN_CondAF_MMDChiSq generator model and synthesizes data.
    Converts one-hot encoded genotypes back to dosage (0, 1, 2).
    """
    print("Starting synthetic data generation with ACGAN_CondAF_MMDChiSq model...")

    # --- Path to the saved generator model ---
    generator_path = os.path.join(MODEL_BASE_PATH, SAVED_GENERATOR_FILENAME)

    if not os.path.exists(generator_path):
        print(f"ERROR: ACGAN_CondAF_MMDChiSq Generator model not found at {generator_path}")
        print("Please check:")
        print(f"  1. MODEL_BASE_PATH ('{MODEL_BASE_PATH}') is correct.")
        print(f"  2. SAVED_GENERATOR_FILENAME ('{SAVED_GENERATOR_FILENAME}') is correct.")
        print(f"  3. The model was successfully saved from training.")
        return

    # --- Load the trained generator ---
    print(f"Loading ACGAN_CondAF_MMDChiSq generator model from: {generator_path}")
    try:
        # compile=False is generally recommended for inference-only models.
        # Custom objects might be needed if the *optimizer's state* (saved with the model)
        # references custom loss functions, even if the generator layers themselves are standard.
        generator = load_model(generator_path, custom_objects=custom_objects, compile=False)
        generator.summary() # Display model structure
    except Exception as e:
        print(f"Error loading ACGAN_CondAF_MMDChiSq generator model: {e}")
        print("If loading fails, try providing dummy versions of custom loss functions used during training")
        print("to the 'custom_objects' dictionary, even if compile=False.")
        return

    # --- Prepare for generation ---
    all_synthetic_genos_dosage = []
    all_synthetic_phenos = []

    # 1. Generate Control Samples (label 0)
    if Num_Contrl > 0:
        print(f"Generating {Num_Contrl} control samples (label 0)...")
        # Create random noise input
        noise_controls = np.random.normal(0, 1, (Num_Contrl, NOISE_DIM))
        # Create integer labels (shape [batch, 1] for this generator's label input)
        labels_controls_input = np.zeros((Num_Contrl, 1), dtype='int32') # Label 0

        # Generate synthetic data (one-hot encoded probabilities)
        # Generator expects [noise, label_input]
        synthetic_controls_one_hot_probs = generator.predict([noise_controls, labels_controls_input], verbose=0)

        # Convert one-hot probabilities back to dosage (0, 1, 2)
        # np.argmax finds the index of the highest probability in the one-hot vector
        synthetic_controls_dosage = np.argmax(synthetic_controls_one_hot_probs, axis=-1)

        all_synthetic_genos_dosage.append(synthetic_controls_dosage)
        all_synthetic_phenos.append(np.zeros(Num_Contrl, dtype=int)) # Phenotype is 0
        print(f"Generated {synthetic_controls_dosage.shape[0]} control genotypes (dosage format).")

    # 2. Generate Case Samples (label 1)
    if Num_Case > 0:
        print(f"Generating {Num_Case} case samples (label 1)...")
        noise_cases = np.random.normal(0, 1, (Num_Case, NOISE_DIM))
        labels_cases_input = np.ones((Num_Case, 1), dtype='int32') # Label 1

        synthetic_cases_one_hot_probs = generator.predict([noise_cases, labels_cases_input], verbose=0)
        synthetic_cases_dosage = np.argmax(synthetic_cases_one_hot_probs, axis=-1)

        all_synthetic_genos_dosage.append(synthetic_cases_dosage)
        all_synthetic_phenos.append(np.ones(Num_Case, dtype=int)) # Phenotype is 1
        print(f"Generated {synthetic_cases_dosage.shape[0]} case genotypes (dosage format).")

    # --- Combine and Save ---
    if not all_synthetic_genos_dosage:
        print("No samples were generated (Num_Contrl and Num_Case might be zero). Exiting.")
        return

    final_synthetic_genos_dosage = np.vstack(all_synthetic_genos_dosage)
    final_synthetic_phenos = np.concatenate(all_synthetic_phenos)

    # Ensure the output directory exists
    output_dir_geno = Path(GENO_OUTPUT_FILE_ACGAN_MMD).parent
    output_dir_pheno = Path(PHENO_OUTPUT_FILE_ACGAN_MMD).parent
    output_dir_geno.mkdir(parents=True, exist_ok=True)
    if output_dir_geno != output_dir_pheno:
         output_dir_pheno.mkdir(parents=True, exist_ok=True)


    print(f"Saving {final_synthetic_genos_dosage.shape[0]} synthetic genotypes to {GENO_OUTPUT_FILE_ACGAN_MMD}...")
    np.savetxt(GENO_OUTPUT_FILE_ACGAN_MMD, final_synthetic_genos_dosage, fmt='%d', delimiter=' ')

    print(f"Saving {final_synthetic_phenos.shape[0]} synthetic phenotypes to {PHENO_OUTPUT_FILE_ACGAN_MMD}...")
    np.savetxt(PHENO_OUTPUT_FILE_ACGAN_MMD, final_synthetic_phenos, fmt='%d', delimiter=' ')

    print("ACGAN_CondAF_MMDChiSq synthetic data generation and saving complete.")
    print(f"Synthetic genotypes shape (dosage): {final_synthetic_genos_dosage.shape}")
    print(f"Synthetic phenotypes shape: {final_synthetic_phenos.shape}")
    print(f"Files saved: ./{GENO_OUTPUT_FILE_ACGAN_MMD}, ./{PHENO_OUTPUT_FILE_ACGAN_MMD}")

if __name__ == '__main__':
    # Optional: GPU memory growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e: print(e)

    generate_synthetic_data_acgan_mmd()
