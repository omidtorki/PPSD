import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import time

# --- Configuration ---
SYNTHETIC_GENO_FILE = '../data_split/geno_synth.txt'
MEMBER_GENO_FILE = '../data_split/member_geno.txt'
NON_MEMBER_GENO_FILE = '../data_split/non_member_geno.txt'

INPUT_DIM = 10000
LATENT_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
LR = 0.0002
NUM_EPOCHS = 30
EVAL_EPOCH_INTERVAL = 5
TEST_SAMPLE_COUNT = 100
RANDOM_STATE_MIA = 42

# --- Model Definitions ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, INPUT_DIM), nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z) * 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM * 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# --- Evaluation Function ---
def evaluate_attack(discriminator, test_data, attack_true_labels, n_members_in_test, device):
    discriminator.eval()
    with torch.no_grad():
        predictions_scores = discriminator(test_data.to(device)).cpu().numpy().flatten()
        sorted_indices = np.argsort(predictions_scores)[::-1]
        attack_predicted_labels = np.zeros_like(attack_true_labels)
        top_n_indices = sorted_indices[:n_members_in_test]
        attack_predicted_labels[top_n_indices] = 1
        
        tn, fp, fn, tp = confusion_matrix(attack_true_labels, attack_predicted_labels, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    discriminator.train()
    return accuracy, tp, tn, fp, fn

# --- Main Attack Logic ---
def run_epoch_based_blackbox_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading attacker's training data (synthetic dataset)...")
    synthetic_data = np.loadtxt(SYNTHETIC_GENO_FILE, dtype=np.float32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_data)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Loading and preparing fixed, random test set...")
    member_data = np.loadtxt(MEMBER_GENO_FILE, dtype=np.float32)
    non_member_data = np.loadtxt(NON_MEMBER_GENO_FILE, dtype=np.float32)
    
    np.random.seed(RANDOM_STATE_MIA)
    member_indices = np.random.choice(len(member_data), TEST_SAMPLE_COUNT, replace=False)
    non_member_indices = np.random.choice(len(non_member_data), TEST_SAMPLE_COUNT, replace=False)
    
    test_members = torch.from_numpy(member_data[member_indices])
    test_non_members = torch.from_numpy(non_member_data[non_member_indices])
    test_data = torch.cat([test_members, test_non_members], dim=0)
    attack_true_labels = np.concatenate([np.ones(TEST_SAMPLE_COUNT), np.zeros(TEST_SAMPLE_COUNT)])

    # 2. Initialize Models and Optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 3. Epoch-Based Training Loop
    # Correctly defined dictionary
    results = {'epochs': [], 'accuracy': [], 'tp': [], 'tn': [], 'fp': [], 'fn': []}
    
    print("\n--- Starting Black-Box Attack Model Training (Epoch-Based, Rank-Based) ---")
    
    print("Evaluating attack at Epoch 0...")
    acc, tp, tn, fp, fn = evaluate_attack(discriminator, test_data, attack_true_labels, TEST_SAMPLE_COUNT, device)
    results['epochs'].append(0)
    results['accuracy'].append(acc)
    # Corrected key 'fn'
    results['tp'].append(tp); results['tn'].append(tn); results['fp'].append(fp); results['fn'].append(fn)
    print(f"Epoch: 0 | MIA Accuracy: {acc:.4f}")

    total_start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        for i, (real_samples,) in enumerate(train_loader):
            optimizer_d.zero_grad()
            real_samples = real_samples.to(device)
            real_labels = torch.ones(real_samples.size(0), 1).to(device)
            real_output = discriminator(real_samples)
            d_loss_real = criterion(real_output, real_labels)
            
            noise = torch.randn(real_samples.size(0), LATENT_DIM).to(device)
            fake_samples = generator(noise)
            fake_labels = torch.zeros(real_samples.size(0), 1).to(device)
            fake_output = discriminator(fake_samples.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            gen_output = discriminator(fake_samples)
            g_loss = criterion(gen_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Attack Model Epoch {epoch}/{NUM_EPOCHS} completed in {time.time() - epoch_start_time:.2f}s.")
        
        if epoch % EVAL_EPOCH_INTERVAL == 0:
            print(f"--- Evaluating MIA at Epoch {epoch} ---")
            acc, tp, tn, fp, fn = evaluate_attack(discriminator, test_data, attack_true_labels, TEST_SAMPLE_COUNT, device)
            results['epochs'].append(epoch)
            results['accuracy'].append(acc)
            # Corrected key 'fn'
            results['tp'].append(tp); results['tn'].append(tn); results['fp'].append(fp); results['fn'].append(fn)
            print(f"Epoch: {epoch} | MIA Accuracy: {acc:.4f} [TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}]")

    print(f"\n--- Training Finished in {time.time() - total_start_time:.2f}s ---")

    # 4. Save and Plot Results
    results_filename = "blackbox_epoch_rank_attack_results.csv"
    # Corrected key 'fn'
    output_data = np.array([
        results['epochs'], results['accuracy'], 
        results['tp'], results['tn'], results['fp'], results['fn']
    ]).T
    np.savetxt(
        results_filename, output_data, delimiter=',',
        header='Epoch,Accuracy,TP,TN,FP,FN', comments='',
        fmt=['%d', '%.4f', '%d', '%d', '%d', '%d']
    )
    print(f"Full results saved to {results_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(results['epochs'], results['accuracy'], marker='o', linestyle='-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (50%)')
    plt.title('Black-Box Attack Performance vs. Training Epoch (Rank-Based)')
    plt.xlabel('Training Epoch of Attack Model')
    plt.ylabel('Attack Accuracy')
    plt.xticks(results['epochs'])
    plt.grid(True)
    plt.legend()
    plt.ylim(0.4, 1.0)
    
    plot_filename = "blackbox_epoch_rank_performance_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    run_epoch_based_blackbox_attack()
