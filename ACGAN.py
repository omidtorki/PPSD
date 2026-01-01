import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Dense, Conv1D, Flatten, Reshape,
                                     Conv1DTranspose, BatchNormalization,
                                     LeakyReLU, Embedding, Concatenate, Multiply, Dropout, UpSampling1D) # Added UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean as MeanMetric
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
# from IPython import display # Not used in this version
import gc

# --- MMD Specific Imports (None extra needed if implemented within TF) ---

# --- Environment and GPU Setup ---
print("TF Version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e: print(e)
else: print("No GPU detected, running on CPU.")

# --- Constants and Parameters ---
USER_GENO_PATH = "geno_10000_2.txt"
USER_LABEL_PATH = "pheno.txt"
CHECKPOINT_DIR = ".training_checkpoints/UserSNP_ACGAN_CondAF_MMDChiSqLoss"

GENOTYPE_LEN = 10000
CHANNELS = 3
NUM_CLASSES = 2
GENOTYPE_SHAPE = (GENOTYPE_LEN, CHANNELS)
NOISE_DIM = 100

LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 5e-5
LABEL_SMOOTHING_FACTOR = 0.9
NOISE_STDDEV_D = 0.01 # Kept this for discriminator input noise
EPOCHS = 100
BATCH_SIZE = 512 # Keep large, adjust if OOM
SAVE_INTERVAL = 10

# Loss Weights
G_ADV_WEIGHT = 1.0
G_CLF_WEIGHT = 1.0
G_COND_AF_WEIGHT = 1.0 
G_MMD_CHI_SQ_WEIGHT = 1.0

# MMD Kernel parameters (RBF kernel)
MMD_SIGMA_LIST = [0.1, 1.0, 10.0] # List of sigma values for multi-kernel MMD, or a single value

# --- Data Loading Function ---
def load_user_data(geno_path, label_path): # Removed snp_indices_path as it wasn't used
    if not os.path.exists(label_path): raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = np.loadtxt(label_path, dtype=int)
    if np.max(labels) >= NUM_CLASSES or np.min(labels) < 0:
        raise ValueError(f"Labels should be 0 or 1. Found min {np.min(labels)}, max {np.max(labels)}.")
    n_samples = labels.shape[0]
    if not os.path.exists(geno_path): raise FileNotFoundError(f"Genotype file not found: {geno_path}")
    print(f"Loading {n_samples} samples from {geno_path}, using all {GENOTYPE_LEN} SNPs...")
    dataset = np.loadtxt(geno_path, dtype=np.float32)
    if dataset.shape[1] != GENOTYPE_LEN:
        print(f"Warning: Genotype file has {dataset.shape[1]} SNPs, but GENOTYPE_LEN is {GENOTYPE_LEN}. Adjusting GENOTYPE_LEN or data.")

    print("Finished loading SNPs.")
    nan_mask = np.isnan(dataset)
    if np.any(nan_mask):
        print(f"Warning: Filling {np.sum(nan_mask)} NaN values with dosage 0.0.")
        dataset[nan_mask] = 0.0
    dataset_int = dataset.astype(np.int32)
    min_val, max_val = np.min(dataset_int), np.max(dataset_int)
    print(f"Dosage range check: Min={min_val}, Max={max_val}")
    if min_val < 0 or max_val > 2:
        raise ValueError(f"Input geno.txt must contain dosages 0, 1, or 2. Found range [{min_val}, {max_val}].")
    print(f"One-hot encoding dosages to {CHANNELS} channels...")
    X_train_one_hot = to_categorical(dataset_int, num_classes=CHANNELS)
    print(f"Final data shapes: X={X_train_one_hot.shape}, Y={labels.shape}")
    return X_train_one_hot.astype(np.float32), labels.astype(np.int32)

# --- Model Definitions (Generator and Discriminator) ---
def make_generator_model():
    noise = layers.Input(shape=(NOISE_DIM,), name='noise_input')
    label = layers.Input(shape=(1,), name='label_input', dtype='int32')
    label_embedding = layers.Flatten()(layers.Embedding(NUM_CLASSES, NOISE_DIM)(label))
    model_input = layers.multiply([noise, label_embedding])
    start_steps = GENOTYPE_LEN // 4
    start_filters = 256
    x = layers.Dense(start_steps * start_filters, use_bias=False, kernel_initializer=HeNormal())(model_input)
    x = layers.Reshape((start_steps, start_filters))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(256, 7, strides=1, padding='same', use_bias=False, kernel_initializer=HeNormal())(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(128, 5, strides=1, padding='same', use_bias=False, kernel_initializer=HeNormal())(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(CHANNELS, 5, strides=1, padding='same', activation='softmax', kernel_initializer=HeNormal())(x)
    return Model([noise, label], x, name='generator')

def make_discriminator_model():
    genotype = layers.Input(shape=GENOTYPE_SHAPE, name='genotype_input')
    label = layers.Input(shape=(1,), name='label_input', dtype='int32')
    embedding_dim_discriminator = GENOTYPE_LEN * CHANNELS
    label_embedding = layers.Flatten()(layers.Embedding(NUM_CLASSES, embedding_dim_discriminator)(label))
    flat_genotype = layers.Flatten()(genotype)
    model_input = layers.multiply([flat_genotype, label_embedding])
    model_input = layers.Reshape(GENOTYPE_SHAPE)(model_input)
    x = layers.LeakyReLU()(model_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 5, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 7, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(256, 5, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_initializer=HeNormal())(x) # Dense layer before outputs
    x = layers.LeakyReLU()(x)
    out_adv = layers.Dense(1, activation='sigmoid', name='adversarial_output')(x)
    out_clf = layers.Dense(1, activation='sigmoid', name='classifier_output')(x) # Sigmoid for binary class
    return Model([genotype, label], [out_adv, out_clf], name='discriminator')

# --- TF Function for Expected Conditional Alternate Allele Freq ---
@tf.function
def calculate_exp_cond_alt_freq_tf(probs, labels):
    n_snps = tf.shape(probs)[1]
    default_freq = tf.zeros((n_snps,), dtype=tf.float32)
    exp_alt_allele_dosage_contribution = 1.0 * probs[..., 1] + 2.0 * probs[..., 2]
    case_mask = tf.equal(labels, 1)
    dosage_case = tf.boolean_mask(exp_alt_allele_dosage_contribution, case_mask, axis=0)
    n_case = tf.cast(tf.shape(dosage_case)[0], tf.float32)
    exp_freq_alt_case = tf.cond(n_case > 0.5, lambda: tf.reduce_mean(dosage_case, axis=0) / 2.0, lambda: default_freq)
    control_mask = tf.equal(labels, 0)
    dosage_control = tf.boolean_mask(exp_alt_allele_dosage_contribution, control_mask, axis=0)
    n_control = tf.cast(tf.shape(dosage_control)[0], tf.float32)
    exp_freq_alt_control = tf.cond(n_control > 0.5, lambda: tf.reduce_mean(dosage_control, axis=0) / 2.0, lambda: default_freq)
    return exp_freq_alt_case, exp_freq_alt_control

# --- TensorFlow Function for Chi-squared Calculation  ---
@tf.function
def calculate_chi_squared_tf(genotype_batch, labels_batch):
    n_snps = tf.shape(genotype_batch)[1]
    control_mask = tf.equal(labels_batch, 0)
    case_mask = tf.equal(labels_batch, 1)
    controls_data = tf.boolean_mask(genotype_batch, control_mask, axis=0)
    cases_data = tf.boolean_mask(genotype_batch, case_mask, axis=0)
    n_controls_batch = tf.cast(tf.shape(controls_data)[0], tf.float32)
    n_cases_batch = tf.cast(tf.shape(cases_data)[0], tf.float32)
    chi_sq_values = tf.zeros(n_snps, dtype=tf.float32)
    if n_controls_batch < 1.0 or n_cases_batch < 1.0: return chi_sq_values
    exp_alt_control = tf.reduce_sum(controls_data, axis=0)
    exp_ref_control = 2.0 * n_controls_batch - exp_alt_control
    exp_alt_case = tf.reduce_sum(cases_data, axis=0)
    exp_ref_case = 2.0 * n_cases_batch - exp_alt_case
    total_ref = exp_ref_control + exp_ref_case
    total_alt = exp_alt_control + exp_alt_case
    total_alleles_batch = (n_controls_batch + n_cases_batch) * 2.0 # Correct total alleles in batch
    epsilon = K.epsilon()
    E_RC_C = (total_ref * (n_controls_batch * 2.0)) / (total_alleles_batch + epsilon)
    E_AC_C = (total_alt * (n_controls_batch * 2.0)) / (total_alleles_batch + epsilon)
    E_RC_CA = (total_ref * (n_cases_batch * 2.0)) / (total_alleles_batch + epsilon)
    E_AC_CA = (total_alt * (n_cases_batch * 2.0)) / (total_alleles_batch + epsilon)
    chi_sq_term1 = tf.square(exp_ref_control - E_RC_C) / (E_RC_C + epsilon)
    chi_sq_term2 = tf.square(exp_alt_control - E_AC_C) / (E_AC_C + epsilon)
    chi_sq_term3 = tf.square(exp_ref_case - E_RC_CA) / (E_RC_CA + epsilon)
    chi_sq_term4 = tf.square(exp_alt_case - E_AC_CA) / (E_AC_CA + epsilon)
    chi_sq_values = chi_sq_term1 + chi_sq_term2 + chi_sq_term3 + chi_sq_term4
    chi_sq_values = tf.where(tf.math.is_finite(chi_sq_values), chi_sq_values, tf.zeros_like(chi_sq_values))
    return chi_sq_values

# --- MMD Loss Implementation ---
@tf.function
def rbf_kernel(x, y, sigma_list):
    """
    Computes RBF kernel between x and y for multiple sigmas.
    x, y: Tensors of shape (batch_size, num_features). num_features is GENOTYPE_LEN (num_snps) here.
    sigma_list: list of sigma values for the RBF kernel.
    Returns a scalar MMD^2 value.
    """
    # Ensure x and y are 2D: (num_samples, num_features=1) for MMD on 1D data (chi-sq values)
    # Chi-squared values are 1D vectors (n_snps,). Reshape for pairwise distances.
    # x and y will be [n_snps, 1]
    x = tf.expand_dims(x, axis=-1) # [n_snps, 1]
    y = tf.expand_dims(y, axis=-1) # [n_snps, 1]

    # Pairwise squared Euclidean distances.
    # XX = x @ x.T (not needed directly, use broadcasting)
    # YY = y @ y.T
    # XY = x @ y.T
    
    # K.sum(K.square(x), axis=-1) gives squared norm of each row vector in x
    # K.expand_dims(..., axis=1) makes it a column vector
    # K.expand_dims(..., axis=0) makes it a row vector
    # dist_sq = K.sum(K.square(x), axis=-1, keepdims=True) - 2 * K.batch_dot(x, K.permute_dimensions(y,(1,0))) + K.transpose(K.sum(K.square(y), axis=-1, keepdims=True))
    # This is for (batch, features). For (features, 1) it's simpler:
    
    x_sqnorms = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) # [n_snps, 1]
    y_sqnorms = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True) # [n_snps, 1]

    # x_sqnorms_row = tf.transpose(x_sqnorms) # [1, n_snps]
    y_sqnorms_col = y_sqnorms # [n_snps, 1]


    # Pairwise squared distances: dist_ij = ||x_i - y_j||^2
    # For x, y of shape [N, 1], [M, 1]
    # dist_sq(x_i, y_j) = (x_i - y_j)^2 = x_i^2 - 2*x_i*y_j + y_j^2
    
    # K(x, x)
    K_XX_dist_sq = x_sqnorms - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(x_sqnorms) # [n_snps, n_snps]
    # K(y, y)
    K_YY_dist_sq = y_sqnorms - 2 * tf.matmul(y, y, transpose_b=True) + tf.transpose(y_sqnorms) # [n_snps, n_snps]
    # K(x, y)
    K_XY_dist_sq = x_sqnorms - 2 * tf.matmul(x, y, transpose_b=True) + tf.transpose(y_sqnorms) # [n_snps, n_snps]


    total_kernel_val_xx = tf.zeros_like(K_XX_dist_sq[0,0]) # scalar
    total_kernel_val_yy = tf.zeros_like(K_YY_dist_sq[0,0]) # scalar
    total_kernel_val_xy = tf.zeros_like(K_XY_dist_sq[0,0]) # scalar

    for sigma_val in sigma_list:
        gamma = 1.0 / (2 * sigma_val**2)
        total_kernel_val_xx += tf.exp(-gamma * K_XX_dist_sq)
        total_kernel_val_yy += tf.exp(-gamma * K_YY_dist_sq)
        total_kernel_val_xy += tf.exp(-gamma * K_XY_dist_sq)
        
    return total_kernel_val_xx, total_kernel_val_yy, total_kernel_val_xy

@tf.function
def mmd_loss_tf(x, y, sigma_list=MMD_SIGMA_LIST):
    """
    Computes MMD^2 between x and y distributions using RBF kernel.
    x, y: Tensors representing samples from two distributions.
          Here, these will be the Chi-squared values (1D vectors of shape [n_snps]).
    sigma_list: list of sigma values for multi-kernel MMD.
    """
    # If x or y are empty (can happen if a batch has only one class for chi-sq calc)
    if tf.size(x) == 0 or tf.size(y) == 0:
        return 0.0 # Or a large constant if this scenario should be penalized. 0.0 means no MMD diff.

    K_XX_sum, K_YY_sum, K_XY_sum = rbf_kernel(x, y, sigma_list)

    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    # For empirical estimate, use means of kernel evaluations
    nx = tf.cast(tf.shape(x)[0], tf.float32)
    ny = tf.cast(tf.shape(y)[0], tf.float32)

    # Sum of kernel values (excluding diagonal for K_XX and K_YY if unbiased, but often simpler to include)
    # For biased estimator (simpler):
    mmd2 = (tf.reduce_sum(K_XX_sum) / (nx * nx) +
            tf.reduce_sum(K_YY_sum) / (ny * ny) -
            2 * tf.reduce_sum(K_XY_sum) / (nx * ny))
    
    # Ensure non-negative (due to numerical precision, mmd2 can sometimes be tiny negative)
    mmd2 = tf.maximum(0.0, mmd2)
    return mmd2

# --- Loss Functions ---
bce = BinaryCrossentropy()

def discriminator_loss(real_adv_output, real_clf_output, fake_adv_output, fake_clf_output, real_labels, fake_labels):
    real_adv_loss = bce(tf.ones_like(real_adv_output) * LABEL_SMOOTHING_FACTOR, real_adv_output)
    fake_adv_loss = bce(tf.zeros_like(fake_adv_output) + (1.0 - LABEL_SMOOTHING_FACTOR), fake_adv_output)
    total_adv_loss = real_adv_loss + fake_adv_loss
    real_labels_float = tf.cast(tf.expand_dims(real_labels, axis=-1), tf.float32)
    fake_labels_float = tf.cast(tf.expand_dims(fake_labels, axis=-1), tf.float32) # ensure this is fake_labels
    real_clf_loss = bce(real_labels_float, real_clf_output)
    fake_clf_loss = bce(fake_labels_float, fake_clf_output)
    total_clf_loss = real_clf_loss + fake_clf_loss
    total_loss = total_adv_loss + total_clf_loss
    return total_loss, total_adv_loss, total_clf_loss

def generator_loss_fn(fake_adv_output, fake_clf_output, target_gen_labels,
                      gen_output_probs,
                      freq_alt_real_control_target_tensor, freq_alt_real_case_target_tensor,
                      real_batch_chi_sq_values # Chi-squared values from the REAL batch
                      ):
    adv_loss = bce(tf.ones_like(fake_adv_output), fake_adv_output)
    target_labels_float = tf.cast(tf.expand_dims(target_gen_labels, axis=-1), tf.float32)
    clf_loss = bce(target_labels_float, fake_clf_output)

    exp_freq_alt_case_gen, exp_freq_alt_control_gen = calculate_exp_cond_alt_freq_tf(gen_output_probs, target_gen_labels)
    mask_control = ~tf.math.is_nan(freq_alt_real_control_target_tensor)
    control_af_loss = tf.cond(tf.size(tf.boolean_mask(freq_alt_real_control_target_tensor, mask_control)) > 0,
                              lambda: tf.reduce_mean(tf.square(tf.boolean_mask(exp_freq_alt_control_gen, mask_control) - tf.boolean_mask(freq_alt_real_control_target_tensor, mask_control))),
                              lambda: 0.0)
    mask_case = ~tf.math.is_nan(freq_alt_real_case_target_tensor)
    case_af_loss = tf.cond(tf.size(tf.boolean_mask(freq_alt_real_case_target_tensor, mask_case)) > 0,
                           lambda: tf.reduce_mean(tf.square(tf.boolean_mask(exp_freq_alt_case_gen, mask_case) - tf.boolean_mask(freq_alt_real_case_target_tensor, mask_case))),
                           lambda: 0.0)
    cond_af_loss = (control_af_loss + case_af_loss) / 2.0

    # --- MMD Chi-squared Loss ---
    gen_exp_dosages = gen_output_probs[..., 1] * 1.0 + gen_output_probs[..., 2] * 2.0
    fake_batch_chi_sq_values = calculate_chi_squared_tf(gen_exp_dosages, target_gen_labels)
    
    # MMD loss between the distributions of Chi-squared values
    # real_batch_chi_sq_values and fake_batch_chi_sq_values are 1D tensors of shape [n_snps]
    mmd_chi_sq_loss_val = mmd_loss_tf(real_batch_chi_sq_values, fake_batch_chi_sq_values)
    
    total_loss_weighted = (G_ADV_WEIGHT * adv_loss +
                           G_CLF_WEIGHT * clf_loss +
                           G_COND_AF_WEIGHT * cond_af_loss +
                           G_MMD_CHI_SQ_WEIGHT * mmd_chi_sq_loss_val)

    return total_loss_weighted, adv_loss, clf_loss, cond_af_loss, mmd_chi_sq_loss_val

# --- Optimizers ---
generator_optimizer = Adam(learning_rate=LEARNING_RATE_G, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=LEARNING_RATE_D, beta_1=0.5)

# --- Training Step ---
@tf.function
def train_step(generator, discriminator, real_genotypes_one_hot, real_labels,
               freq_alt_real_control_tensor, freq_alt_real_case_tensor):
    current_batch_size = tf.shape(real_genotypes_one_hot)[0] # Get actual batch size

    noise_d = tf.random.normal([current_batch_size, NOISE_DIM])
    fake_target_condition_labels_d = tf.random.uniform([current_batch_size], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    fake_target_condition_labels_d_exp = tf.expand_dims(fake_target_condition_labels_d, axis=-1)
    real_labels_exp = tf.expand_dims(real_labels, axis=-1)

    with tf.GradientTape() as disc_tape:
        generated_probs_d = generator([noise_d, fake_target_condition_labels_d_exp], training=True)
        real_genotypes_noisy = real_genotypes_one_hot + tf.random.normal(tf.shape(real_genotypes_one_hot), stddev=NOISE_STDDEV_D)
        fake_genotypes_noisy = generated_probs_d + tf.random.normal(tf.shape(generated_probs_d), stddev=NOISE_STDDEV_D)
        real_genotypes_noisy = tf.clip_by_value(real_genotypes_noisy, 0.0, 1.0)
        fake_genotypes_noisy = tf.clip_by_value(fake_genotypes_noisy, 0.0, 1.0)
        real_adv_output, real_clf_output = discriminator([real_genotypes_noisy, real_labels_exp], training=True)
        fake_adv_output, fake_clf_output = discriminator([fake_genotypes_noisy, fake_target_condition_labels_d_exp], training=True)
        disc_loss, disc_adv_loss, disc_clf_loss = discriminator_loss(
            real_adv_output, real_clf_output, fake_adv_output, fake_clf_output,
            real_labels, fake_target_condition_labels_d)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    noise_g = tf.random.normal([current_batch_size, NOISE_DIM]) # Use actual batch size
    gen_target_condition_labels_g = tf.random.uniform([current_batch_size], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    gen_target_condition_labels_g_exp = tf.expand_dims(gen_target_condition_labels_g, axis=-1)

    with tf.GradientTape() as gen_tape:
        gen_output_probs_g = generator([noise_g, gen_target_condition_labels_g_exp], training=True)
        fake_adv_output_g, fake_clf_output_g = discriminator([gen_output_probs_g, gen_target_condition_labels_g_exp], training=True)
        
        real_batch_dosages = tf.cast(tf.argmax(real_genotypes_one_hot, axis=-1), tf.float32)
        real_batch_chi_sq_values_for_loss = calculate_chi_squared_tf(real_batch_dosages, real_labels)

        gen_total_loss_w, gen_adv_uw, gen_clf_uw, gen_cond_af_uw, gen_mmd_chi_sq_uw = generator_loss_fn(
            fake_adv_output_g, fake_clf_output_g, gen_target_condition_labels_g,
            gen_output_probs_g,
            freq_alt_real_control_tensor, freq_alt_real_case_tensor,
            real_batch_chi_sq_values_for_loss)
    gradients_of_generator = gen_tape.gradient(gen_total_loss_w, generator.trainable_variables)
    if gradients_of_generator is None or any(g is None for g in gradients_of_generator):
        tf.print("Warning: None gradients for generator.")
        # To prevent error, return zero for generator losses if update is skipped
        gen_total_unweighted_sum=tf.constant(0.0); gen_adv_uw=tf.constant(0.0); gen_clf_uw=tf.constant(0.0); gen_cond_af_uw=tf.constant(0.0); gen_mmd_chi_sq_uw=tf.constant(0.0)
    else:
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        gen_total_unweighted_sum = gen_adv_uw + gen_clf_uw + gen_cond_af_uw + gen_mmd_chi_sq_uw

    return (disc_loss, disc_adv_loss, disc_clf_loss,
            gen_total_unweighted_sum, gen_adv_uw, gen_clf_uw, gen_cond_af_uw, gen_mmd_chi_sq_uw)

# --- Pre-calculation of Target Statistics (NumPy) ---
def precompute_cond_alt_freq_targets(X_real_dosage, Y_real_labels):
    print("Pre-computing target conditional alternate allele frequencies...")
    n_samples, n_snps = X_real_dosage.shape
    target_freq_control = np.full(n_snps, np.nan, dtype=np.float32)
    target_freq_case = np.full(n_snps, np.nan, dtype=np.float32)
    control_mask = (Y_real_labels == 0); case_mask = (Y_real_labels == 1)
    X_control = X_real_dosage[control_mask]; X_case = X_real_dosage[case_mask]
    n_control = X_control.shape[0]; n_case = X_case.shape[0]
    print(f"Calculating freqs for {n_control} controls and {n_case} cases...")
    if n_control > 0:
        alt_counts_control = np.sum(X_control == 1, axis=0) * 1.0 + np.sum(X_control == 2, axis=0) * 2.0
        target_freq_control = alt_counts_control / (n_control * 2.0)
    if n_case > 0:
        alt_counts_case = np.sum(X_case == 1, axis=0) * 1.0 + np.sum(X_case == 2, axis=0) * 2.0
        target_freq_case = alt_counts_case / (n_case * 2.0)
    target_freq_control = np.clip(target_freq_control, 0.0, 1.0)
    target_freq_case = np.clip(target_freq_case, 0.0, 1.0)
    print(f"Target Control Alt Freq calculated ({np.sum(~np.isnan(target_freq_control))} valid).")
    print(f"Target Case Alt Freq calculated ({np.sum(~np.isnan(target_freq_case))} valid).")
    return target_freq_control, target_freq_case

# --- Training Loop ---
def train(dataset_X, dataset_Y, generator, discriminator,
          target_freq_control, target_freq_case, epochs, batch_size_param):
    freq_alt_real_control_tensor = tf.convert_to_tensor(target_freq_control, dtype=tf.float32)
    freq_alt_real_case_tensor = tf.convert_to_tensor(target_freq_case, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset_X, dataset_Y))\
                                  .shuffle(dataset_X.shape[0])\
                                  .batch(batch_size_param, drop_remainder=True)\
                                  .prefetch(tf.data.experimental.AUTOTUNE)
    n_steps_per_epoch = dataset_X.shape[0] // batch_size_param
    n_total_steps = n_steps_per_epoch * epochs
    print(f"\n--- Starting AC-GAN Training with Cond AF Loss and MMD Chi-squared Loss ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size_param}, Steps/Epoch: {n_steps_per_epoch}")
    print(f"G Loss Weights: Adv={G_ADV_WEIGHT}, Clf={G_CLF_WEIGHT}, CondAF={G_COND_AF_WEIGHT}, MMDChiSq={G_MMD_CHI_SQ_WEIGHT}")

    d_loss_m = MeanMetric(name='d_loss'); d_adv_m = MeanMetric(name='d_adv'); d_clf_m = MeanMetric(name='d_clf')
    g_loss_m = MeanMetric(name='g_loss'); g_adv_m = MeanMetric(name='g_adv'); g_clf_m = MeanMetric(name='g_clf')
    g_cond_af_m = MeanMetric(name='g_cond_af'); g_mmd_chi_sq_m = MeanMetric(name='g_mmd_chi_sq')
    
    all_metrics = [d_loss_m, d_adv_m, d_clf_m, g_loss_m, g_adv_m, g_clf_m, g_cond_af_m, g_mmd_chi_sq_m]
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    current_step = 0; start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for metric in all_metrics: metric.reset_state()
        for step, (real_batch_X, real_batch_y) in enumerate(train_dataset):
            current_step += 1
            losses = train_step(generator, discriminator, real_batch_X, real_batch_y,
                                freq_alt_real_control_tensor, freq_alt_real_case_tensor)
            d_loss, d_adv, d_clf, g_loss, g_adv, g_clf, g_cond_af, g_mmd_chi_sq = losses
            
            d_loss_m(d_loss); d_adv_m(d_adv); d_clf_m(d_clf)
            g_loss_m(g_loss); g_adv_m(g_adv); g_clf_m(g_clf); g_cond_af_m(g_cond_af); g_mmd_chi_sq_m(g_mmd_chi_sq)

            if current_step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {current_step}/{n_total_steps} "
                      f"D:{d_loss_m.result():.3f}(A:{d_adv_m.result():.3f},C:{d_clf_m.result():.3f}) "
                      f"G:{g_loss_m.result():.3f}(A:{g_adv_m.result():.3f},C:{g_clf_m.result():.3f},AF:{g_cond_af_m.result():.4f},MMDChiSq:{g_mmd_chi_sq_m.result():.4f}) "
                      f"Time:{elapsed:.1f}s")

        print(f"\n--- Epoch {epoch + 1}/{epochs} Completed ---")
        print(f"Time: {time.time() - epoch_start_time:.1f}s "
              f"Avg D Loss:{d_loss_m.result():.3f} (A:{d_adv_m.result():.3f}, C:{d_clf_m.result():.3f}) "
              f"Avg G Loss:{g_loss_m.result():.3f} (A:{g_adv_m.result():.3f}, C:{g_clf_m.result():.3f}, CondAF:{g_cond_af_m.result():.4f}, MMDChiSq:{g_mmd_chi_sq_m.result():.4f})")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            g_model_path = os.path.join(CHECKPOINT_DIR, f'generator_epoch_{epoch+1}.keras')
            d_model_path = os.path.join(CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1}.keras')
            generator.save(g_model_path); discriminator.save(d_model_path)
            print(f"Models saved for epoch {epoch+1} to {CHECKPOINT_DIR}")

    print("--- Training Finished ---")
    generator.save('ACGAN_CondAF_MMDChiSq_generator_final.keras')
    discriminator.save('ACGAN_CondAF_MMDChiSq_discriminator_final.keras')
    print("Final models saved.")

# --- Main Execution ---
if __name__ == '__main__':
    print("Loading and preparing user data...")
    X_train, Y_train = None, None
    try:
        X_train, Y_train = load_user_data(USER_GENO_PATH, USER_LABEL_PATH)
    except Exception as e: print(f"Error loading data: {e}"); exit()
    if X_train is None: print("Error: Data loading failed. Exiting."); exit()

    X_real_dosage_for_targets = np.argmax(X_train, axis=-1)
    target_freq_control, target_freq_case = precompute_cond_alt_freq_targets(X_real_dosage_for_targets, Y_train)
    del X_real_dosage_for_targets; gc.collect()

    print("Building models...")
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    print("--- Generator Summary ---"); generator.summary(line_length=120)
    print("\n--- Discriminator Summary ---"); discriminator.summary(line_length=120)

    print("\nStarting training...")
    train(X_train, Y_train, generator, discriminator,
          target_freq_control, target_freq_case,
          epochs=EPOCHS, batch_size_param=BATCH_SIZE)
    print("\nScript finished.")
