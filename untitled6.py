# -*- coding: utf-8 -*-
"""
Python script for analyzing isothermal degradation kinetics using a
Physics-Informed Neural Operator (PINO) with an FNO backbone.
Models: dy/dt = K(T) * y^n * (1-y)^m, where K(T) = A * exp(-Ea / (R*T)).

Steps:
1.  Loads experimental data (Purity vs. Time vs. Temp) from Excel.
2.  Prepares data: Kelvin conversion, Conversion y calculation.
3.  Defines a common time grid and interpolates experimental y onto this grid
    for each condition (T).
4.  Defines a PINO model (FNO1d) taking normalized T as an input channel
    over the time grid and outputting y on the grid.
5.  Treats A, Ea, n, m as learnable parameters.
6.  Defines loss: Data MSE (on grid) + Physics ODE residual MSE + IC MSE.
7.  Trains the PINO and parameters using PyTorch and Adam.
8.  Reports learned A, Ea, n, m.
9.  Generates predictions using the trained PINO for evaluation and extrapolation.
10. Calculates evaluation metrics (MAE, RMSE, R2, MAPE) based on Purity.
11. Generates plots: Arrhenius, Residuals, Predictions vs. Data.
12. Saves extrapolated predictions to Excel.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import time
import os
import warnings

# Try importing neuraloperator
try:
    from neuralop.models import FNO1d
except ImportError:
    print("ERROR: neuraloperator library not found.")
    print("Please install it: pip install neuraloperator torch_harmonics")
    exit()

# Suppress warnings (e.g., from interpolation)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Configuration
# =============================================================================
excel_filename = 'purity_data.xlsx'
predictions_filename = 'purity_predictions_pino_fno_3yr.xlsx'
prediction_days = 3 * 365 # 3 years

# --- PINO & Training Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FNO Hyperparameters
fno_modes = 16        # Number of Fourier modes
fno_width = 64        # Hidden channel dimension
n_layers = 4          # Number of FNO layers

# Time Grid Configuration
time_grid_points = 128 # Number of points for the FNO's time domain discretization
# Ensure grid covers prediction range
max_time_domain = prediction_days * 1.05

# Training Hyperparameters
learning_rate = 1e-3
num_epochs = 10000 # PINO might need fewer epochs than PINN, adjust as needed
batch_size = 8      # Number of conditions per batch
physics_loss_weight = 0.05 # Weight for the PDE residual loss (adjust based on convergence)
initial_condition_weight = 1.0 # Weight for y(0)=0 loss
print_interval = 200 # Print progress every N epochs
early_stopping_patience = 500 # Stop if validation loss doesn't improve

# Parameter bounds/initial guesses (same as PINN example)
initial_log_A_guess = np.log(1e10)
initial_log_Ea_guess = np.log(80000)
initial_n_guess = 0.5
initial_m_guess = 1.0
n_max_bound = 4.0
m_max_bound = 4.0

# Gas constant (J/mol/K)
R = 8.314

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =============================================================================
# 1. Data Loading and Basic Preparation (Similar to previous scripts)
# =============================================================================
try:
    df_all = pd.read_excel(excel_filename)
    print(f"--- Successfully loaded data from '{excel_filename}' ---")
    print(f"Found {len(df_all)} total data points.")
except FileNotFoundError:
    print(f"ERROR: Excel file '{excel_filename}' not found.")
    # Attempt to create dummy data if file not found
    print("Attempting to create dummy data 'purity_data.xlsx'...")
    try:
        # --- Create Dummy Data --- (Reusing from PINN example)
        data = {
            'Experiment': ['BatchA'] * 12 + ['BatchB'] * 12,
            'Temperature_C': ([25] * 4 + [40] * 4 + [50] * 4) * 2,
            'Time_days': ([0, 30, 60, 90] * 3) * 2,
            'Purity': [ # Example data simulating degradation
                100.0, 99.5, 99.0, 98.5, 100.0, 98.0, 96.0, 94.0, 100.0, 96.0, 92.0, 88.0, # BatchA
                99.8, 99.3, 98.8, 98.3, 99.8, 97.8, 95.8, 93.8, 99.8, 95.8, 91.8, 87.8, # BatchB
            ]
        }
        df_dummy = pd.DataFrame(data)
        df_dummy['Purity'] += np.random.randn(len(df_dummy)) * 0.3 # Add noise
        extra_points = pd.DataFrame({
             'Experiment': ['BatchA', 'BatchB'], 'Temperature_C': [40, 40],
             'Time_days': [180, 180], 'Purity': [90.0 + np.random.randn()*0.3, 89.8 + np.random.randn()*0.3]
        })
        df_all = pd.concat([df_dummy, extra_points], ignore_index=True)
        df_all.to_excel(excel_filename, index=False)
        print(f"Dummy data created and saved to '{excel_filename}'. Please review it.")
        print(f"Found {len(df_all)} total data points.")
        # --- End Create Dummy Data ---
    except Exception as create_e:
        print(f"ERROR: Could not create dummy data: {create_e}")
        exit()
except Exception as e:
    print(f"ERROR: Could not read Excel file '{excel_filename}': {e}")
    exit()

# Basic Data Cleaning & Preparation
df_all = df_all.dropna(subset=['Experiment', 'Temperature_C', 'Time_days', 'Purity'])
df_all = df_all.astype({'Temperature_C': float, 'Time_days': float, 'Purity': float})
df_all = df_all[df_all['Purity'] >= 0]

# Convert Temperature to Kelvin
df_all['Temperature_K'] = df_all['Temperature_C'] + 273.15

# Calculate Purity_t0 and Fraction_Remaining (Robust calculation)
df_all['Time_rank'] = df_all.groupby(['Experiment', 'Temperature_C'])['Time_days'].rank(method='first', ascending=True)
purity_t0_map = df_all[df_all['Time_rank'] == 1].set_index(['Experiment', 'Temperature_C'])['Purity']
df_all['Purity_t0'] = df_all.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map.get)
if df_all['Purity_t0'].isnull().any():
     min_time_purity = df_all.loc[df_all.groupby(['Experiment', 'Temperature_C'])['Time_days'].idxmin()]
     purity_t0_map_min = min_time_purity.set_index(['Experiment', 'Temperature_C'])['Purity']
     df_all['Purity_t0'] = df_all['Purity_t0'].fillna(df_all.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map_min.get))
df_all['Purity_t0'] = df_all.groupby(['Experiment', 'Temperature_C'])['Purity_t0'].ffill().bfill()
df_all.dropna(subset=['Purity_t0'], inplace=True)
df_all = df_all[df_all['Purity_t0'] > 1e-6]
df_all['Fraction_Remaining'] = (df_all['Purity'] / df_all['Purity_t0']).clip(upper=1.0, lower=0.0)

# Calculate Conversion y (alpha)
df_all['Conversion_y'] = 1.0 - df_all['Fraction_Remaining']
df_all['Conversion_y'] = df_all['Conversion_y'].clip(lower=0.0, upper=0.999999)

print(f"Prepared data columns: {df_all.columns.tolist()}")
print(f"Number of data points after preparation: {len(df_all)}")
print("-" * 40)

# =============================================================================
# 2. Prepare Data for FNO (Interpolation onto Common Grid)
# =============================================================================
print("--- Preparing Data for FNO ---")

# Normalize Temperature (Kelvin)
T_min_k = df_all['Temperature_K'].min()
T_max_k = df_all['Temperature_K'].max()

def normalize_temp(T_k):
    if (T_max_k - T_min_k) < 1e-6: return 0.0
    return (T_k - T_min_k) / (T_max_k - T_min_k)

df_all['Temp_K_norm'] = df_all['Temperature_K'].apply(normalize_temp)

# Define the common time grid for FNO input/output
time_grid = torch.linspace(0, max_time_domain, time_grid_points, dtype=torch.float32).to(device)
time_step = time_grid[1] - time_grid[0] # For gradient calculation

# Group data by condition (Experiment + Temperature)
grouped_conditions = df_all.groupby(['Experiment', 'Temperature_K'])

fno_data = [] # List to store tuples: (T_k_norm, T_k, time_grid_np, y_on_grid)

for name, group in grouped_conditions:
    experiment, temp_k = name
    temp_k_norm = group['Temp_K_norm'].iloc[0]
    purity_t0 = group['Purity_t0'].iloc[0] # Needed later for purity calc

    # Sort data by time for interpolation
    group = group.sort_values('Time_days')
    time_exp = group['Time_days'].values
    y_exp = group['Conversion_y'].values

    # Ensure y(0) = 0 for interpolation
    if 0.0 not in time_exp:
        time_exp = np.insert(time_exp, 0, 0.0)
        y_exp = np.insert(y_exp, 0, 0.0)
    else:
        # Ensure the y value at t=0 is exactly 0
        y_exp[time_exp == 0.0] = 0.0

    # Interpolate y onto the common time_grid
    # Use linear interpolation, fill beyond bounds with edge values (or 0 for start, last value for end)
    if len(time_exp) > 1:
        interp_func = interp1d(time_exp, y_exp, kind='linear', bounds_error=False,
                               fill_value=(y_exp[0], y_exp[-1])) # Extrapolate flat
        y_on_grid = interp_func(time_grid.cpu().numpy())
    elif len(time_exp) == 1: # Only t=0 point
         y_on_grid = np.zeros(time_grid_points) # Assume no degradation if only t=0
    else: # No data for this condition? Skip.
        print(f"Warning: Skipping condition {name} due to insufficient data for interpolation.")
        continue

    # Ensure interpolated values are valid
    y_on_grid = np.clip(y_on_grid, 0.0, 1.0)

    fno_data.append({
        'condition_name': name,
        'T_k_norm': temp_k_norm,
        'T_k': temp_k,
        'Purity_t0': purity_t0,
        'y_target_on_grid': torch.tensor(y_on_grid, dtype=torch.float32).to(device)
    })

print(f"Processed {len(fno_data)} unique conditions for FNO input.")

# Split conditions into training and validation sets
train_data, val_data = train_test_split(fno_data, test_size=0.2, random_state=42)

print(f"Training conditions: {len(train_data)}")
print(f"Validation conditions: {len(val_data)}")
print("-" * 40)

# --- Create PyTorch Dataset and DataLoader ---
class DegradationDataset(Dataset):
    def __init__(self, data_list, time_grid_tensor):
        self.data = data_list
        self.time_grid = time_grid_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        T_k_norm = item['T_k_norm']
        y_target = item['y_target_on_grid']

        # Create FNO input: replicate T_k_norm across the time grid as a channel
        # Shape: (1, grid_size) -> (channels, grid_size)
        fno_input = torch.full_like(self.time_grid, T_k_norm).unsqueeze(0) # Add channel dim

        return {
            'input': fno_input,         # Shape: (1, time_grid_points)
            'target': y_target,         # Shape: (time_grid_points)
            'T_k_norm': T_k_norm,       # Scalar normalized temp
            'T_k': item['T_k']          # Scalar original temp (Kelvin)
        }

train_dataset = DegradationDataset(train_data, time_grid)
val_dataset = DegradationDataset(val_data, time_grid)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 3. Define PINO Model (FNO-based)
# =============================================================================

class PINO_FNO(nn.Module):
    def __init__(self, modes, width, n_layers, time_grid_points,
                 initial_log_A, initial_log_Ea, initial_n, initial_m, n_max, m_max):
        super().__init__()
        self.modes = modes
        self.width = width
        self.time_grid_points = time_grid_points

        # FNO1d model:
        # in_channels=1 (T_k_norm), out_channels=1 (y)
        # n_modes=(modes,)
        # hidden_channels=width
        # n_layers=n_layers
        self.fno = FNO1d(n_modes_height=modes, hidden_channels=width,
                         in_channels=1, out_channels=1, n_layers=n_layers)

        # --- Learnable Physical Parameters --- (Same as PINN)
        self.log_A = nn.Parameter(torch.tensor([initial_log_A], dtype=torch.float32))
        self.log_Ea = nn.Parameter(torch.tensor([initial_log_Ea], dtype=torch.float32))

        def inv_sigmoid(p):
            p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
            return torch.log(p / (1.0 - p))

        self.logit_n = nn.Parameter(torch.tensor([inv_sigmoid(torch.tensor(initial_n / n_max))], dtype=torch.float32))
        self.logit_m = nn.Parameter(torch.tensor([inv_sigmoid(torch.tensor(initial_m / m_max))], dtype=torch.float32))

        self.n_max = torch.tensor(n_max, dtype=torch.float32).to(device)
        self.m_max = torch.tensor(m_max, dtype=torch.float32).to(device)

    def forward(self, x_input):
        # x_input shape: (batch_size, 1, time_grid_points) - contains T_k_norm replicated
        y_pred_grid = self.fno(x_input) # Output shape: (batch_size, 1, time_grid_points)
        # Apply sigmoid activation to ensure output y is between 0 and 1
        return torch.sigmoid(y_pred_grid.squeeze(1)) # Return shape: (batch_size, time_grid_points)

    def get_params(self):
        A = torch.exp(self.log_A)
        Ea = torch.exp(self.log_Ea)
        n = torch.sigmoid(self.logit_n) * self.n_max
        m = torch.sigmoid(self.logit_m) * self.m_max
        return A, Ea, n, m

# --- Instantiate the Model ---
pino_model = PINO_FNO(fno_modes, fno_width, n_layers, time_grid_points,
                      initial_log_A_guess, initial_log_Ea_guess,
                      initial_n_guess, initial_m_guess, n_max_bound, m_max_bound).to(device)

print("PINO-FNO Model Architecture:")
print(pino_model)
# print(f"Number of parameters: {sum(p.numel() for p in pino_model.parameters() if p.requires_grad)}") # Optional: check model size
A_init, Ea_init, n_init, m_init = pino_model.get_params()
print("\nInitial parameter guesses (actual values):")
print(f"  A: {A_init.item():.4e} day^-1")
print(f"  Ea: {Ea_init.item() / 1000:.2f} kJ/mol")
print(f"  n: {n_init.item():.3f}")
print(f"  m: {m_init.item():.3f}")
print("-" * 40)


# =============================================================================
# 4. Define Loss Function (Data + Physics + IC)
# =============================================================================

def pino_loss_fn(model, batch, time_grid_tensor, time_step_val, physic_weight, ic_weight):
    """Combined loss function for PINO-FNO."""
    fno_input = batch['input'].to(device)
    y_target = batch['target'].to(device)
    T_k_norm_batch = batch['T_k_norm'].to(device) # Shape: (batch_size)
    T_k_batch = batch['T_k'].to(device)           # Shape: (batch_size)

    # 1. Forward pass - Get predicted y on the grid
    y_pred_grid = model(fno_input) # Shape: (batch_size, time_grid_points)

    # 2. Data Loss (MSE on the grid)
    data_loss = nn.functional.mse_loss(y_pred_grid, y_target)

    # 3. Physics Loss (ODE Residual)
    # Calculate dy/dt using finite difference (torch.gradient)
    # Need gradient along the time dimension (dim=1)
    dy_dt = torch.gradient(y_pred_grid, spacing=(time_step_val,), dim=1)[0]

    # Get current physical parameters
    A, Ea, n, m = model.get_params()

    # Calculate K(T) for each item in the batch
    # Need to reshape T_k_batch to (batch_size, 1) to broadcast with grid
    K = A * torch.exp(-Ea / (R * T_k_batch.unsqueeze(1))) # Shape: (batch_size, 1)

    # Calculate ODE RHS: K * y^n * (1-y)^m
    epsilon = 1e-10
    y_safe = torch.clamp(y_pred_grid, epsilon, 1.0 - epsilon)

    # Ensure parameters n, m broadcast correctly with y_safe
    n_b = n.expand_as(y_safe)
    m_b = m.expand_as(y_safe)
    term1 = torch.pow(y_safe, n_b)
    term2 = torch.pow(1.0 - y_safe, m_b)

    ode_rhs = K * term1 * term2 # K broadcasts correctly

    # Calculate residual
    residual = dy_dt - ode_rhs
    phys_loss = torch.mean(residual**2)

    # 4. Initial Condition Loss (y(t=0) = 0)
    # Extract prediction at the first time point (index 0)
    y_at_t0 = y_pred_grid[:, 0]
    ic_loss = nn.functional.mse_loss(y_at_t0, torch.zeros_like(y_at_t0))

    # Total Loss
    total_loss = data_loss + physic_weight * phys_loss + ic_weight * ic_loss
    return total_loss, data_loss, phys_loss, ic_loss


# =============================================================================
# 5. Training Loop
# =============================================================================
optimizer = Adam(pino_model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150, verbose=True)

best_val_loss = float('inf')
epochs_no_improve = 0
training_start_time = time.time()

print("--- Starting PINO-FNO Training ---")
for epoch in range(num_epochs):
    pino_model.train()
    train_loss_epoch = 0.0
    train_data_loss_epoch = 0.0
    train_phys_loss_epoch = 0.0
    train_ic_loss_epoch = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        loss, data_loss, phys_loss, ic_loss = pino_loss_fn(
            pino_model, batch, time_grid, time_step.item(),
            physics_loss_weight, initial_condition_weight
        )
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        train_data_loss_epoch += data_loss.item()
        train_phys_loss_epoch += phys_loss.item()
        train_ic_loss_epoch += ic_loss.item()

    # Average training losses
    avg_train_loss = train_loss_epoch / len(train_loader)
    avg_data_loss = train_data_loss_epoch / len(train_loader)
    avg_phys_loss = train_phys_loss_epoch / len(train_loader)
    avg_ic_loss = train_ic_loss_epoch / len(train_loader)

    # Validation step
    pino_model.eval()
    val_loss_epoch = 0.0
    val_data_loss_epoch = 0.0
    with torch.no_grad():
        for batch in val_loader:
            loss, data_loss, phys_loss, ic_loss = pino_loss_fn(
                pino_model, batch, time_grid, time_step.item(),
                physics_loss_weight, initial_condition_weight
            )
            val_loss_epoch += loss.item()
            val_data_loss_epoch += data_loss.item() # Primarily track data loss for validation

    avg_val_loss = val_loss_epoch / len(val_loader)
    avg_val_data_loss = val_data_loss_epoch / len(val_loader)

    # Update learning rate scheduler based on total validation loss
    scheduler.step(avg_val_loss)

    # Print progress
    if epoch % print_interval == 0 or epoch == num_epochs - 1:
        A_curr, Ea_curr, n_curr, m_curr = pino_model.get_params()
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4e} (Data: {avg_data_loss:.4e}, Phys: {avg_phys_loss:.4e}, IC: {avg_ic_loss:.4e}) | "
              f"Val Loss: {avg_val_loss:.4e} (Data: {avg_val_data_loss:.4e}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
              f"A: {A_curr.item():.3e}, Ea: {Ea_curr.item()/1000:.2f}, n: {n_curr.item():.3f}, m: {m_curr.item():.3f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model state dictionary
        torch.save(pino_model.state_dict(), 'best_pino_fno_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"\nEarly stopping triggered at epoch {epoch} due to no improvement in validation loss for {early_stopping_patience} epochs.")
        break

training_duration = time.time() - training_start_time
print(f"--- Training Finished in {training_duration:.2f} seconds ---")

# Load the best model state
print("Loading best model state based on validation loss.")
# --- MODIFICATION START ---
# Load the state dictionary with weights_only=False for compatibility
# Also add map_location=device for robustness
try:
    pino_model.load_state_dict(torch.load('best_pino_fno_model.pth', map_location=device, weights_only=False))
except FileNotFoundError:
    print("Warning: 'best_pino_fno_model.pth' not found. Using the model state from the end of training.")
except Exception as e:
    print(f"Warning: Could not load best model state dict: {e}. Using the model state from the end of training.")
# --- MODIFICATION END ---
pino_model.eval() # Ensure model is in eval mode after loading or if loading failed

# =============================================================================
# 6. Report Learned Parameters and Arrhenius Plot
# =============================================================================
print("\n--- Learned Parameters (PINO-FNO) ---")
A_final, Ea_final, n_final, m_final = pino_model.get_params()
A_final_val = A_final.item()
Ea_final_val = Ea_final.item()
n_final_val = n_final.item()
m_final_val = m_final.item()

print(f"  Activation Energy (Ea): {Ea_final_val / 1000:.2f} kJ/mol")
print(f"  Pre-exponential Factor (A): {A_final_val:.4e} day^-1")
print(f"  Reaction Order (n): {n_final_val:.4f}")
print(f"  Reaction Order (m): {m_final_val:.4f}")
print("-" * 40)

# --- Plot Learned Arrhenius Relationship --- (Identical to PINN plot)
print("--- Generating Arrhenius Plot (Learned Relationship) ---")
temps_k_plot = np.linspace(T_min_k, T_max_k, 50)
inv_T_plot = 1.0 / temps_k_plot
with torch.no_grad():
    K_pred_plot = A_final_val * np.exp(-Ea_final_val / (R * temps_k_plot))
ln_K_pred_plot = np.log(K_pred_plot)
unique_temps_k_exp = sorted(df_all['Temperature_K'].unique())
inv_T_exp = 1.0 / np.array(unique_temps_k_exp)
with torch.no_grad():
    K_pred_exp = A_final_val * np.exp(-Ea_final_val / (R * np.array(unique_temps_k_exp)))
ln_K_pred_exp = np.log(K_pred_exp)

plt.figure(figsize=(7, 5))
plt.scatter(inv_T_exp, ln_K_pred_exp, marker='o', s=80, facecolors='none', edgecolors='blue',
            label='ln(K) at Exp Temps (Learned Params)')
plt.plot(inv_T_plot, ln_K_pred_plot, 'r-',
         label=f'Learned Arrhenius Fit (Ea={Ea_final_val/1000:.1f} kJ/mol)')
plt.xlabel("1 / Temperature (1/K)")
plt.ylabel("ln(K)  (K in day^-1)")
plt.title("Learned Arrhenius Relationship (PINO-FNO)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
print("-" * 40)


# =============================================================================
# 7. Generate Predictions for Evaluation and Extrapolation
# =============================================================================
print("--- Generating Predictions using Trained PINO-FNO ---")
df_eval = df_all.copy()

# --- Predictions for Existing Experimental Points ---
# We need to evaluate the PINO at the specific experimental times,
# not just on the grid used for training. We do this by running the PINO
# for each condition and then interpolating its output grid to the specific times.

df_eval['Predicted_Conversion_y'] = np.nan # Initialize column

print("Evaluating model at experimental time points...")
pino_model.eval()
with torch.no_grad():
    for condition_data in fno_data: # Use fno_data which has all conditions
        name = condition_data['condition_name']
        T_k_norm = condition_data['T_k_norm']
        purity_t0 = condition_data['Purity_t0']
        experiment, temp_k = name

        # Prepare input for this single condition
        fno_input_cond = torch.full_like(time_grid, T_k_norm).unsqueeze(0).unsqueeze(0).to(device) # Batch=1, Chan=1

        # Get PINO prediction on the full time grid
        y_pred_on_grid_cond = pino_model(fno_input_cond).squeeze().cpu().numpy() # Shape (grid_points,)

        # Create an interpolation function from the PINO's output grid
        interp_pino_output = interp1d(time_grid.cpu().numpy(), y_pred_on_grid_cond,
                                      kind='linear', bounds_error=False,
                                      fill_value=(y_pred_on_grid_cond[0], y_pred_on_grid_cond[-1]))

        # Find the original experimental points for this condition
        condition_mask = (df_eval['Experiment'] == experiment) & (df_eval['Temperature_K'] == temp_k)
        exp_times_cond = df_eval.loc[condition_mask, 'Time_days'].values

        # Interpolate to get predictions at experimental times
        if len(exp_times_cond) > 0: # Ensure there are times to predict for
            y_pred_at_exp_times = interp_pino_output(exp_times_cond)
            y_pred_at_exp_times = np.clip(y_pred_at_exp_times, 0.0, 1.0)

            # Store predictions back in the evaluation dataframe
            df_eval.loc[condition_mask, 'Predicted_Conversion_y'] = y_pred_at_exp_times
        else:
             print(f"Warning: No experimental times found for condition {name} in df_eval.")


# Calculate predicted purity
# Ensure Purity_t0 is aligned correctly before calculation
df_eval = pd.merge(df_eval.drop(columns=['Purity_t0'], errors='ignore'),
                   df_all[['Experiment', 'Temperature_K', 'Time_days', 'Purity_t0']].drop_duplicates(),
                   on=['Experiment', 'Temperature_K', 'Time_days'], how='left')

df_eval['Predicted_Purity'] = (1.0 - df_eval['Predicted_Conversion_y']) * df_eval['Purity_t0']
df_eval = df_eval.dropna(subset=['Predicted_Purity', 'Purity', 'Predicted_Conversion_y']) # Drop rows where prediction failed

print(f"Generated predictions for {len(df_eval)} experimental points.")

# --- Extrapolation Predictions (up to prediction_days) ---
# The PINO already predicts on the full time_grid up to max_time_domain
prediction_results_export = []
print(f"Extracting extrapolated predictions up to {prediction_days} days...")
pino_model.eval()
with torch.no_grad():
     for condition_data in fno_data:
        name = condition_data['condition_name']
        T_k_norm = condition_data['T_k_norm']
        temp_k = condition_data['T_k']
        purity_t0 = condition_data['Purity_t0']
        experiment, _ = name

        # Prepare input
        fno_input_cond = torch.full_like(time_grid, T_k_norm).unsqueeze(0).unsqueeze(0).to(device)
        # Get prediction on grid
        y_pred_on_grid_cond = pino_model(fno_input_cond).squeeze().cpu().numpy()

        # Store results for export
        k_pred_cond = A_final_val * np.exp(-Ea_final_val / (R * temp_k))
        time_grid_np = time_grid.cpu().numpy()

        for t, y_p in zip(time_grid_np, y_pred_on_grid_cond):
             if t <= prediction_days: # Only save up to the desired prediction time
                p_p = (1.0 - y_p) * purity_t0
                prediction_results_export.append({
                    'Experiment': experiment, 'Temperature_C': temp_k - 273.15, 'Time_days': t,
                    'Predicted_Conversion_y': y_p, 'Predicted_Purity': p_p,
                    'k_pred_day^-1': k_pred_cond, 'n_used': n_final_val, 'm_used': m_final_val
                })

print("Extrapolation finished.")
print("-" * 40)


# =============================================================================
# 8. Calculate Evaluation Metrics (Based on Purity)
# =============================================================================
print("--- Model Evaluation Metrics (Based on Purity Predictions) ---")
# MAPE function (same as before)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true > 1e-6 # Use a small threshold for MAPE stability
    if np.sum(non_zero_idx) == 0: return np.nan
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

# Evaluate on the entire dataset
metrics_results = {}
if len(df_eval) > 0:
    y_true = df_eval['Purity']
    y_pred = df_eval['Predicted_Purity']
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    metrics_results['Overall'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_eval)}
    print(f"\nMetrics for Overall Data (N={len(df_eval)}):")
    print(f"  Mean Absolute Error (MAE):     {mae:.3f} % Purity")
    print(f"  Root Mean Squared Error (RMSE):{rmse:.3f} % Purity")
    print(f"  R-squared (R2):                {r2:.4f}")
    print(f"  Mean Abs Percentage Error (MAPE): {mape:.2f} %")
else:
    print("\nNo data available for evaluation.")
print("-" * 40)

# =============================================================================
# 9. Residual Plot (Based on Purity)
# =============================================================================
print("--- Generating Residual Plot (Based on Purity) ---")
if len(df_eval) > 0:
    df_eval['Residual'] = df_eval['Purity'] - df_eval['Predicted_Purity']
    plt.figure(figsize=(10, 6))
    plt.scatter(df_eval['Predicted_Purity'], df_eval['Residual'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Purity (%)")
    plt.ylabel("Residual (Actual - Predicted Purity %)")
    plt.title("Residual Plot (PINO-FNO Model)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
else:
    print("Skipping residual plot as no evaluation data is available.")
print("-" * 40)


# =============================================================================
# 10. Extrapolation and Prediction Plot
# =============================================================================
print(f"--- Plotting Predictions up to {prediction_days} days ---")

if prediction_results_export:
    df_predictions_plot = pd.DataFrame(prediction_results_export)

    unique_experiments = df_all['Experiment'].unique()
    n_exp = len(unique_experiments)
    n_cols = 2
    n_rows = (n_exp + n_cols - 1) // n_cols
    fig_preds, axes_preds = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    axes_preds_flat = axes_preds.flatten()
    plot_idx = 0

    # Use consistent colors
    unique_temps_plot = sorted(df_all['Temperature_C'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps_plot)))
    temp_color_map = {temp: color for temp, color in zip(unique_temps_plot, colors)}

    for experiment_id in unique_experiments:
        ax = axes_preds_flat[plot_idx]
        exp_data_plot = df_all[df_all['Experiment'] == experiment_id]
        exp_preds_plot = df_predictions_plot[df_predictions_plot['Experiment'] == experiment_id]
        temps_in_exp = sorted(exp_data_plot['Temperature_C'].unique())

        for temp_c in temps_in_exp:
            color = temp_color_map.get(temp_c, 'grey')
            # Plot experimental data
            exp_data_temp = exp_data_plot[exp_data_plot['Temperature_C'] == temp_c]
            if not exp_data_temp.empty:
                ax.scatter(exp_data_temp['Time_days'], exp_data_temp['Purity'],
                           label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)
            # Plot PINO prediction curve (from the saved grid predictions)
            pred_data_temp = exp_preds_plot[exp_preds_plot['Temperature_C'] == temp_c]
            if not pred_data_temp.empty:
                # Sort prediction data just in case
                pred_data_temp = pred_data_temp.sort_values('Time_days')
                ax.plot(pred_data_temp['Time_days'], pred_data_temp['Predicted_Purity'],
                        '--', label=f'{temp_c:.0f}°C (PINO Pred)', color=color)

        # Finalize subplot
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Purity (%)")
        ax.set_title(f"Experiment: {experiment_id} - PINO-FNO Predictions")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.7)
        min_purity_exp = exp_data_plot['Purity'].min() if not exp_data_plot.empty else 0
        max_purity_exp = exp_data_plot['Purity_t0'].max() if not exp_data_plot.empty else 100
        ax.set_ylim(bottom=max(0, min_purity_exp - 10), top=max_purity_exp + 2)
        ax.set_xlim(left=-prediction_days*0.02, right=prediction_days*1.02) # Ensure full range is shown
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes_preds_flat)): fig_preds.delaxes(axes_preds_flat[i])
    fig_preds.suptitle(f"PINO-FNO Model Predictions (Extrapolated to {prediction_days} days)", fontsize=16, y=1.02)
    fig_preds.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
else:
    print("Skipping prediction plot as no extrapolation results were generated.")


# =============================================================================
# 11. Save Predictions to Excel
# =============================================================================
if prediction_results_export:
    df_predictions_save = pd.DataFrame(prediction_results_export)
    df_predictions_save = df_predictions_save.sort_values(by=['Experiment', 'Temperature_C', 'Time_days'])
    try:
        df_predictions_save.to_excel(predictions_filename, index=False, engine='openpyxl')
        print(f"\n--- Predictions saved to '{predictions_filename}' ---")
    except Exception as e:
        print(f"\nERROR: Could not save predictions to Excel file '{predictions_filename}': {e}")
else:
    print("\nNo predictions were generated for export.")

# Clean up saved model file
if os.path.exists('best_pino_fno_model.pth'):
    try:
        os.remove('best_pino_fno_model.pth')
        print("Cleaned up saved model file.")
    except Exception as e:
        print(f"Warning: Could not remove saved model file 'best_pino_fno_model.pth': {e}")


print("-" * 40)
print("PINO-FNO Script finished.")
