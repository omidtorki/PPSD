import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean as MeanMetric
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gc

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

# --- Parameters for Grey-Box Attack ---
USER_GENO_PATH = "data_split/geno_synth.txt"
USER_LABEL_PATH = "data_split/pheno_synth.txt"
MEMBER_GENO_PATH_FOR_MIA = 'data_split/member_geno.txt'
MEMBER_PHENO_PATH_FOR_MIA = 'data_split/member_pheno.txt'
NON_MEMBER_GENO_PATH_FOR_MIA = 'data_split/non_member_geno.txt'
NON_MEMBER_PHENO_PATH_FOR_MIA = 'data_split/non_member_pheno.txt'

GENOTYPE_LEN = 10000
CHANNELS = 3
NUM_CLASSES = 2
GENOTYPE_SHAPE = (GENOTYPE_LEN, CHANNELS)
NOISE_DIM = 100
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 5e-5
LABEL_SMOOTHING_FACTOR = 0.9
NOISE_STDDEV_D = 0.01
G_ADV_WEIGHT = 1.0
G_CLF_WEIGHT = 1.0
G_COND_AF_WEIGHT = 1.0
G_MMD_CHI_SQ_WEIGHT = 1.0
MMD_SIGMA_LIST = [0.1, 1.0, 10.0]

EPOCHS = 30
BATCH_SIZE = 512
EVAL_EPOCH_INTERVAL = 5
TEST_SAMPLE_COUNT_MIA = 100
RANDOM_STATE_MIA = 42

# ==============================================================================
# ALL ORIGINAL FUNCTIONS FROM TRAIN SCRIPT ARE INCLUDED
# ==============================================================================

def load_user_data(geno_path, label_path, one_hot=True):
    if not os.path.exists(label_path): raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = np.loadtxt(label_path, dtype=int)
    if not os.path.exists(geno_path): raise FileNotFoundError(f"Genotype file not found: {geno_path}")
    dataset = np.loadtxt(geno_path, dtype=np.float32)
    if one_hot:
        dataset_int = dataset.astype(np.int32)
        X_train_one_hot = to_categorical(dataset_int, num_classes=CHANNELS)
        return X_train_one_hot.astype(np.float32), labels.astype(np.int32)
    return dataset.astype(np.float32), labels.astype(np.int32)

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
    x = layers.Dense(512, kernel_initializer=HeNormal())(x)
    x = layers.LeakyReLU()(x)
    out_adv = layers.Dense(1, activation='sigmoid', name='adversarial_output')(x)
    out_clf = layers.Dense(1, activation='sigmoid', name='classifier_output')(x)
    return Model([genotype, label], [out_adv, out_clf], name='discriminator')

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
    total_alleles_batch = (n_controls_batch + n_cases_batch) * 2.0
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

@tf.function
def rbf_kernel(x, y, sigma_list):
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)
    x_sqnorms = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    y_sqnorms = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
    K_XX_dist_sq = x_sqnorms - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(x_sqnorms)
    K_YY_dist_sq = y_sqnorms - 2 * tf.matmul(y, y, transpose_b=True) + tf.transpose(y_sqnorms)
    K_XY_dist_sq = x_sqnorms - 2 * tf.matmul(x, y, transpose_b=True) + tf.transpose(y_sqnorms)
    total_kernel_val_xx = tf.zeros_like(K_XX_dist_sq[0,0])
    total_kernel_val_yy = tf.zeros_like(K_YY_dist_sq[0,0])
    total_kernel_val_xy = tf.zeros_like(K_XY_dist_sq[0,0])
    for sigma_val in sigma_list:
        gamma = 1.0 / (2 * sigma_val**2)
        total_kernel_val_xx += tf.exp(-gamma * K_XX_dist_sq)
        total_kernel_val_yy += tf.exp(-gamma * K_YY_dist_sq)
        total_kernel_val_xy += tf.exp(-gamma * K_XY_dist_sq)
    return total_kernel_val_xx, total_kernel_val_yy, total_kernel_val_xy

@tf.function
def mmd_loss_tf(x, y, sigma_list=MMD_SIGMA_LIST):
    if tf.size(x) == 0 or tf.size(y) == 0: return 0.0
    K_XX_sum, K_YY_sum, K_XY_sum = rbf_kernel(x, y, sigma_list)
    nx = tf.cast(tf.shape(x)[0], tf.float32)
    ny = tf.cast(tf.shape(y)[0], tf.float32)
    mmd2 = (tf.reduce_sum(K_XX_sum) / (nx * nx) +
            tf.reduce_sum(K_YY_sum) / (ny * ny) -
            2 * tf.reduce_sum(K_XY_sum) / (nx * ny))
    return tf.maximum(0.0, mmd2)

bce = BinaryCrossentropy()
def discriminator_loss(real_adv_output, real_clf_output, fake_adv_output, fake_clf_output, real_labels, fake_labels):
    real_adv_loss = bce(tf.ones_like(real_adv_output) * LABEL_SMOOTHING_FACTOR, real_adv_output)
    fake_adv_loss = bce(tf.zeros_like(fake_adv_output), fake_adv_output) # No smoothing on fake
    total_adv_loss = real_adv_loss + fake_adv_loss
    real_labels_float = tf.cast(tf.expand_dims(real_labels, axis=-1), tf.float32)
    fake_labels_float = tf.cast(tf.expand_dims(fake_labels, axis=-1), tf.float32)
    real_clf_loss = bce(real_labels_float, real_clf_output)
    fake_clf_loss = bce(fake_labels_float, fake_clf_output)
    total_clf_loss = real_clf_loss + fake_clf_loss
    return total_adv_loss + total_clf_loss

def generator_loss_fn(fake_adv_output, fake_clf_output, target_gen_labels, gen_output_probs,
                      freq_alt_real_control_target_tensor, freq_alt_real_case_target_tensor,
                      real_batch_chi_sq_values):
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
    gen_exp_dosages = gen_output_probs[..., 1] * 1.0 + gen_output_probs[..., 2] * 2.0
    fake_batch_chi_sq_values = calculate_chi_squared_tf(gen_exp_dosages, target_gen_labels)
    mmd_chi_sq_loss_val = mmd_loss_tf(real_batch_chi_sq_values, fake_batch_chi_sq_values)
    total_loss_weighted = (G_ADV_WEIGHT * adv_loss + G_CLF_WEIGHT * clf_loss +
                           G_COND_AF_WEIGHT * cond_af_loss + G_MMD_CHI_SQ_WEIGHT * mmd_chi_sq_loss_val)
    return total_loss_weighted

@tf.function
def train_step(generator, discriminator, real_genotypes_one_hot, real_labels,
               freq_alt_real_control_tensor, freq_alt_real_case_tensor, 
               generator_optimizer, discriminator_optimizer):
    current_batch_size = tf.shape(real_genotypes_one_hot)[0]
    noise_d = tf.random.normal([current_batch_size, NOISE_DIM])
    fake_target_condition_labels_d = tf.random.uniform([current_batch_size], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    fake_target_condition_labels_d_exp = tf.expand_dims(fake_target_condition_labels_d, axis=-1)
    real_labels_exp = tf.expand_dims(real_labels, axis=-1)

    with tf.GradientTape() as disc_tape:
        generated_probs_d = generator([noise_d, fake_target_condition_labels_d_exp], training=True)
        real_adv_output, real_clf_output = discriminator([real_genotypes_one_hot, real_labels_exp], training=True)
        fake_adv_output, fake_clf_output = discriminator([generated_probs_d, fake_target_condition_labels_d_exp], training=True)
        disc_loss = discriminator_loss(real_adv_output, real_clf_output, fake_adv_output, fake_clf_output, real_labels, fake_target_condition_labels_d)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    noise_g = tf.random.normal([current_batch_size, NOISE_DIM])
    gen_target_condition_labels_g = tf.random.uniform([current_batch_size], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    gen_target_condition_labels_g_exp = tf.expand_dims(gen_target_condition_labels_g, axis=-1)

    with tf.GradientTape() as gen_tape:
        gen_output_probs_g = generator([noise_g, gen_target_condition_labels_g_exp], training=True)
        fake_adv_output_g, fake_clf_output_g = discriminator([gen_output_probs_g, gen_target_condition_labels_g_exp], training=True)
        real_batch_dosages = tf.cast(tf.argmax(real_genotypes_one_hot, axis=-1), tf.float32)
        real_batch_chi_sq_values_for_loss = calculate_chi_squared_tf(real_batch_dosages, real_labels)
        gen_loss = generator_loss_fn(fake_adv_output_g, fake_clf_output_g, gen_target_condition_labels_g,
                                     gen_output_probs_g, freq_alt_real_control_tensor, freq_alt_real_case_tensor,
                                     real_batch_chi_sq_values_for_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def precompute_cond_alt_freq_targets(X_real_dosage, Y_real_labels):
    print("Pre-computing target conditional alternate allele frequencies...")
    n_samples, n_snps = X_real_dosage.shape
    target_freq_control = np.full(n_snps, np.nan, dtype=np.float32)
    target_freq_case = np.full(n_snps, np.nan, dtype=np.float32)
    control_mask = (Y_real_labels == 0); case_mask = (Y_real_labels == 1)
    X_control = X_real_dosage[control_mask]; X_case = X_real_dosage[case_mask]
    n_control = X_control.shape[0]; n_case = X_case.shape[0]
    if n_control > 0:
        alt_counts_control = np.sum(X_control == 1, axis=0) * 1.0 + np.sum(X_control == 2, axis=0) * 2.0
        target_freq_control = alt_counts_control / (n_control * 2.0)
    if n_case > 0:
        alt_counts_case = np.sum(X_case == 1, axis=0) * 1.0 + np.sum(X_case == 2, axis=0) * 2.0
        target_freq_case = alt_counts_case / (n_case * 2.0)
    return np.clip(target_freq_control, 0.0, 1.0), np.clip(target_freq_case, 0.0, 1.0)


# --- NEW: Function to Evaluate Membership Inference Attack ---
def evaluate_mia_attack(discriminator, mia_test_X, mia_test_y, mia_true_labels, n_members):
    pred_outputs = discriminator.predict([mia_test_X, mia_test_y], verbose=0)
    pred_scores = pred_outputs[0].flatten()

    sorted_indices = np.argsort(pred_scores)[::-1]
    predicted_labels = np.zeros_like(mia_true_labels)
    predicted_labels[sorted_indices[:n_members]] = 1

    tn, fp, fn, tp = confusion_matrix(mia_true_labels, predicted_labels, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, tp, tn, fp, fn

# --- MODIFIED Training Loop with Integrated MIA Evaluation ---
def train_and_evaluate_attack_loop(dataset_X, dataset_Y, generator, discriminator, target_freq_control, target_freq_case, epochs, batch_size_param):
    generator_optimizer = Adam(learning_rate=LEARNING_RATE_G, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=LEARNING_RATE_D, beta_1=0.5)
    freq_alt_real_control_tensor = tf.convert_to_tensor(target_freq_control, dtype=tf.float32)
    freq_alt_real_case_tensor = tf.convert_to_tensor(target_freq_case, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset_X, dataset_Y))\
                                  .shuffle(dataset_X.shape[0])\
                                  .batch(batch_size_param, drop_remainder=True)\
                                  .prefetch(tf.data.experimental.AUTOTUNE)

    print("\nPreparing fixed, random test set for MIA evaluation...")
    all_member_X, all_member_y = load_user_data(MEMBER_GENO_PATH_FOR_MIA, MEMBER_PHENO_PATH_FOR_MIA, one_hot=True)
    all_non_member_X, all_non_member_y = load_user_data(NON_MEMBER_GENO_PATH_FOR_MIA, NON_MEMBER_PHENO_PATH_FOR_MIA, one_hot=True)
    
    np.random.seed(RANDOM_STATE_MIA)
    member_indices = np.random.choice(len(all_member_X), TEST_SAMPLE_COUNT_MIA, replace=False)
    non_member_indices = np.random.choice(len(all_non_member_X), TEST_SAMPLE_COUNT_MIA, replace=False)
    mia_test_X = np.concatenate((all_member_X[member_indices], all_non_member_X[non_member_indices]))
    mia_test_y = np.concatenate((all_member_y[member_indices].reshape(-1, 1), all_non_member_y[non_member_indices].reshape(-1, 1)))
    mia_true_labels = np.concatenate((np.ones(TEST_SAMPLE_COUNT_MIA), np.zeros(TEST_SAMPLE_COUNT_MIA)))

    print(f"\n--- Starting Grey-Box Attack Training for {epochs} Epochs ---")
    mia_results = {'epochs': [], 'accuracy': [], 'tps': [], 'tns': [], 'fps': [], 'fns': []}
    
    print("Evaluating attack at Epoch 0...")
    acc, tp, tn, fp, fn = evaluate_mia_attack(discriminator, mia_test_X, mia_test_y, mia_true_labels, TEST_SAMPLE_COUNT_MIA)
    mia_results['epochs'].append(0); mia_results['accuracy'].append(acc); mia_results['tps'].append(tp); mia_results['tns'].append(tn); mia_results['fps'].append(fp); mia_results['fns'].append(fn)
    print(f"Epoch: 0 | MIA Accuracy: {acc:.4f}")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        for step, (real_batch_X, real_batch_y) in enumerate(train_dataset):
            train_step(generator, discriminator, real_batch_X, real_batch_y,
                       freq_alt_real_control_tensor, freq_alt_real_case_tensor,
                       generator_optimizer, discriminator_optimizer)
        
        print(f"Attack Model Epoch {epoch}/{epochs} completed in {time.time() - epoch_start_time:.2f}s.")

        if epoch % EVAL_EPOCH_INTERVAL == 0:
            print(f"--- Evaluating MIA at Epoch {epoch} ---")
            acc, tp, tn, fp, fn = evaluate_mia_attack(discriminator, mia_test_X, mia_test_y, mia_true_labels, TEST_SAMPLE_COUNT_MIA)
            mia_results['epochs'].append(epoch); mia_results['accuracy'].append(acc); mia_results['tps'].append(tp); mia_results['tns'].append(tn); mia_results['fps'].append(fp); mia_results['fns'].append(fn)
            print(f"Epoch: {epoch} | MIA Accuracy: {acc:.4f} [TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}]")

    print("\n--- Attack Model Training Finished ---")
    
    results_filename = "greybox_full_script_results.csv"
    results_array = np.array([
        mia_results['epochs'], mia_results['accuracy'], 
        mia_results['tps'], mia_results['tns'], mia_results['fps'], mia_results['fns']
    ]).T
    np.savetxt(results_filename, results_array, delimiter=',', header='Epoch,Accuracy,TP,TN,FP,FN', comments='')
    print(f"Detailed results saved to {results_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(mia_results['epochs'], mia_results['accuracy'], marker='o', linestyle='-')
    plt.title('Grey-Box Attack Performance vs. Training Epoch')
    plt.xlabel('Training Epoch of Attack Model')
    plt.ylabel('Membership Inference Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (50%)')
    plt.xticks(mia_results['epochs'])
    plt.grid(True)
    plt.legend()
    plt.ylim(0.4, 1.05)
    
    plot_filename = "greybox_full_script_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Running Grey-Box Attack Simulation using Original Training Script ---")
    X_attack_train, Y_attack_train = load_user_data(USER_GENO_PATH, USER_LABEL_PATH, one_hot=True)
    
    X_synth_dosage = np.argmax(X_attack_train, axis=-1)
    target_freq_control_synth, target_freq_case_synth = precompute_cond_alt_freq_targets(X_synth_dosage, Y_attack_train)
    del X_synth_dosage; gc.collect()

    generator = make_generator_model()
    discriminator = make_discriminator_model()
    
    print("\n--- Attacker's Discriminator Summary ---")
    discriminator.summary()
    
    train_and_evaluate_attack_loop(
        X_attack_train, Y_attack_train, generator, discriminator,
        target_freq_control_synth, target_freq_case_synth,
        epochs=EPOCHS, batch_size_param=BATCH_SIZE
    )
    print("\nScript finished.")
