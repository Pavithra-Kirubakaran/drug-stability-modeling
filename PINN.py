# -*- coding: utf-8 -*-
"""
Conceptual Python script for predicting drug stability using a
Physics-Informed Neural Network (PINN) with PyTorch.

*** Version 5: Temperature-Dependent n(T) and m(T), Increased lambda_physics ***
*** Set lambda_physics = 10.0 ***


This script outlines the structure for:
1. Loading and preparing data.
2. Defining Neural Networks (NNs):
   - solution_net: Approximates conversion y(t, T).
   - n_net: Predicts reaction order n as a function of T.
   - m_net: Predicts reaction order m as a function of T.
3. Defining learnable parameters for Arrhenius kinetics:
   - logA, logEa (for K = A * exp(-Ea / RT))
4. Defining the PINN loss function:
   - Data Loss: Measures mismatch between NN prediction and experimental data.
   - Physics Loss: Measures how well the NN output satisfies the ODE
     dy/dt = A*exp(-Ea/RT) * y^n(T) * (1-y)^m(T) at collocation points.
   - IC Loss: Enforces y(0, T) = 0.
5. Setting up the training loop using PyTorch's optimizer.
6. Making predictions using the trained networks.
7. Evaluating the model using MAE, RMSE, R2, MAPE.
8. Saving extrapolated predictions to Excel.
9. Comparing PINN dynamics with solve_ivp using learned parameters and networks.

NOTE: This adds complexity. Tuning the architectures of n_net, m_net,
      loss weights, learning rate, and epochs is crucial.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
# Added imports for completion
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import solve_ivp
import warnings

# Suppress potential warnings from metrics calculation if needed
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Configuration & Setup
# =============================================================================
excel_filename = 'purity_data.xlsx' # Input data file
predictions_filename = 'purity_predictions_pinn_autocat_temp_nm.xlsx' # Output file
# PINN Hyperparameters (EXAMPLES - NEED TUNING)
learning_rate = 1e-3
num_epochs = 20000 # May need more epochs due to increased complexity
# *** Increased lambda_physics significantly ***
lambda_physics = 100.0 # Weight for the physics loss component (Increased from 0.1)
lambda_ic = 1.0 # Weight for the initial condition loss (can be tuned)
nn_arch_solution = [2, 50, 50, 50, 1] # Input (t_scaled, T_scaled), Output (y)
# Architectures for temperature-dependent n and m networks
nn_arch_n = [1, 10, 10, 1] # Input (T_scaled), Hidden layers, Output (n)
nn_arch_m = [1, 10, 10, 1] # Input (T_scaled), Hidden layers, Output (m)


max_training_days = 90 # Use data up to this time for training loss
prediction_days = 3 * 365 # Extrapolation time
num_collocation_points = 5000 # Number of points to enforce physics loss

# Physical Constants
R = 8.314 # Gas constant (J/mol/K)
epsilon = 1e-10 # Small value for numerical stability in rate calculation

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. Data Loading and Preparation (Same as before)
# =============================================================================
try:
    df_all = pd.read_excel(excel_filename)
    print(f"--- Successfully loaded data from '{excel_filename}' ---")
except FileNotFoundError:
    print(f"ERROR: Excel file '{excel_filename}' not found.")
    exit()
except Exception as e:
    print(f"ERROR: Could not read Excel file '{excel_filename}': {e}")
    exit()

# Basic Data Cleaning & Preparation
df_all = df_all.dropna(subset=['Experiment', 'Temperature_C', 'Time_days', 'Purity'])
df_all = df_all.astype({'Temperature_C': float, 'Time_days': float, 'Purity': float})
df_all = df_all[df_all['Purity'] >= 0]
df_all['Temperature_K'] = df_all['Temperature_C'] + 273.15

# Calculate Purity_t0 and Fraction_Remaining / Conversion y
df_all['Time_rank'] = df_all.groupby(['Experiment', 'Temperature_C'])['Time_days'].rank(method='first')
purity_t0_map = df_all.loc[df_all['Time_rank'] == 1].set_index(['Experiment', 'Temperature_C'])['Purity']
df_all['Purity_t0'] = df_all.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map)
df_all['Purity_t0'] = df_all['Purity_t0'].fillna(method='ffill').fillna(method='bfill')
df_all = df_all.dropna(subset=['Purity_t0'])
df_all = df_all[df_all['Purity_t0'] > 1e-6]

df_all['Fraction_Remaining'] = (df_all['Purity'] / df_all['Purity_t0']).clip(upper=1.0)
df_all['Conversion_y'] = (1.0 - df_all['Fraction_Remaining']).clip(lower=0.0, upper=0.999999)

# Separate Training Data (for data loss)
df_train_data = df_all[
    (df_all['Time_days'] > 0) &
    (df_all['Time_days'] <= max_training_days) &
    df_all['Conversion_y'].notna()
].copy()

print(f"Number of training data points (for data loss): {len(df_train_data)}")
if len(df_train_data) == 0:
    print("ERROR: No data available for training loss.")
    exit()

# --- Prepare Data for PyTorch ---
# Scaling inputs (t, T)
t_scale_factor = max_training_days
T_min_K = df_all['Temperature_K'].min()
T_max_K = df_all['Temperature_K'].max()
T_scale_factor = T_max_K # Using simple scaling by max K

# Training data tensors
t_data = torch.tensor(df_train_data['Time_days'].values, dtype=torch.float32).unsqueeze(1)
T_data = torch.tensor(df_train_data['Temperature_K'].values, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(df_train_data['Conversion_y'].values, dtype=torch.float32).unsqueeze(1)

t_data_scaled = t_data / t_scale_factor
T_data_scaled = T_data / T_scale_factor

X_data = torch.cat((t_data_scaled, T_data_scaled), dim=1).to(device)
y_data = y_data.to(device)

# Collocation points
t_colloc_vals = np.random.uniform(0, max_training_days, num_collocation_points) # Sample within training time range
T_colloc_vals = np.random.uniform(T_min_K, T_max_K, num_collocation_points)

# Move T_colloc (unscaled T) to the correct device
T_colloc = torch.tensor(T_colloc_vals, dtype=torch.float32).unsqueeze(1).to(device)
# Scale t_colloc_vals and move to device
t_colloc_scaled = torch.tensor(t_colloc_vals / t_scale_factor, dtype=torch.float32).unsqueeze(1).to(device)
# Scale T_colloc (already on device)
T_colloc_scaled = T_colloc / T_scale_factor

# X_colloc contains scaled t and scaled T
X_colloc = torch.cat((t_colloc_scaled, T_colloc_scaled), dim=1).to(device)
X_colloc.requires_grad_(True)

# Initial condition points (t=0)
t_zero = torch.zeros(num_collocation_points // 10, 1, dtype=torch.float32)
T_at_zero_vals = np.random.uniform(T_min_K, T_max_K, num_collocation_points // 10)
# Move T_at_zero to the correct device
T_at_zero = torch.tensor(T_at_zero_vals, dtype=torch.float32).unsqueeze(1).to(device)

t_zero_scaled = t_zero / t_scale_factor
T_at_zero_scaled = T_at_zero / T_scale_factor
# Move t_zero_scaled to device before concatenating
t_zero_scaled = t_zero_scaled.to(device)

X_zero = torch.cat((t_zero_scaled, T_at_zero_scaled), dim=1).to(device)


# =============================================================================
# 2. Define Neural Network Architecture & Learnable Parameters
# =============================================================================

class FeedForwardNN(nn.Module):
    """Simple Feed Forward Neural Network."""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        # Use Tanh or SiLU activation
        # self.activation = nn.Tanh()
        self.activation = nn.SiLU() # Swish activation often works well
        layer_list = []
        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(self.activation)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

# NN for the solution y(t, T)
solution_net = FeedForwardNN(nn_arch_solution).to(device)
# NN for reaction order n(T)
n_net = FeedForwardNN(nn_arch_n).to(device)
# NN for reaction order m(T)
m_net = FeedForwardNN(nn_arch_m).to(device)


# Learnable kinetic parameters (A, Ea only)
initial_logA = np.log(1e10) # Adjust initial guesses if needed
initial_logEa = np.log(83140.0)

logA = nn.Parameter(torch.tensor(initial_logA, dtype=torch.float32, device=device))
logEa = nn.Parameter(torch.tensor(initial_logEa, dtype=torch.float32, device=device))
# n_param and m_param are REMOVED


# =============================================================================
# 3. Define PINN Loss Function
# =============================================================================

def pinn_loss_autocat_temp_nm(X_data, y_data, X_colloc, T_colloc_unscaled, X_zero,
                              solution_net, logA, logEa, n_net, m_net, # Pass networks
                              lambda_physics, lambda_ic, t_scale_factor):
    """Calculates the combined PINN loss for the autocatalytic model
       with temperature-dependent n and m."""
    T_colloc_unscaled = T_colloc_unscaled.to(X_colloc.device)

    # --- Data Loss ---
    y_pred_data = solution_net(X_data)
    y_pred_data = torch.relu(y_pred_data) # Ensure y >= 0
    loss_data = torch.mean((y_pred_data - y_data)**2)

    # --- Physics Loss ---
    # Input for n_net, m_net is scaled Temperature
    T_colloc_scaled_input = X_colloc[:, 1:2] # Extract scaled T column

    y_pred_colloc = solution_net(X_colloc)
    y_pred_colloc = torch.relu(y_pred_colloc).clamp(max=1.0) # Ensure 0 <= y <= 1

    dy_dt_scaled = grad(y_pred_colloc, X_colloc,
                        grad_outputs=torch.ones_like(y_pred_colloc),
                        create_graph=True)[0][:, 0:1]

    dy_dt_unscaled = dy_dt_scaled / (1.0 / t_scale_factor)

    # Get fixed parameters A, Ea
    A = torch.exp(logA)
    Ea = torch.exp(logEa)

    # Calculate n(T) and m(T) using the networks
    n = torch.relu(n_net(T_colloc_scaled_input)) # Ensure n >= 0
    m = torch.relu(m_net(T_colloc_scaled_input)) # Ensure m >= 0

    # Calculate K(T) using Arrhenius (use unscaled Temperature)
    K_pred = A * torch.exp(-Ea / (R * T_colloc_unscaled))

    # Calculate rate: K * y^n(T) * (1-y)^m(T)
    y_clipped = torch.clamp(y_pred_colloc, epsilon, 1.0 - epsilon)
    # Need to handle potential 0^0 if n/m can be 0 and y is 0 or 1
    # Using torch.pow is safer for non-integer exponents
    term1 = torch.pow(y_clipped, n)
    term2 = torch.pow(1.0 - y_clipped, m)

    f_pred = K_pred * term1 * term2
    f_pred = torch.relu(f_pred) # Ensure rate is non-negative

    residual = dy_dt_unscaled - f_pred
    loss_physics = torch.mean(residual**2)

    # --- Initial Condition Loss (y(0, T) = 0) ---
    y_pred_zero = solution_net(X_zero)
    y_pred_zero = torch.relu(y_pred_zero)
    loss_ic = torch.mean(y_pred_zero**2)

    # --- Combined Loss ---
    total_loss = loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic

    # Return parameters for monitoring (A, Ea only now)
    return total_loss, A, Ea

# =============================================================================
# 4. Training Setup
# =============================================================================
# Combine parameters from all networks and fixed parameters
all_params = (
    list(solution_net.parameters()) +
    list(n_net.parameters()) +
    list(m_net.parameters()) +
    [logA, logEa]
)
optimizer = torch.optim.Adam(all_params, lr=learning_rate)
# Optional: Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# =============================================================================
# 5. Training Loop
# =============================================================================
print("\n--- Starting PINN Training (Autocatalytic Form, Temp-Dependent n, m) ---")
start_time = time.time()
loss_history = []
param_history = {'A': [], 'Ea': []} # Only A, Ea are single values now
loss_comp_history = {'data': [], 'physics': [], 'ic': []}


for epoch in range(num_epochs):
    solution_net.train()
    n_net.train()
    m_net.train()

    def closure():
        optimizer.zero_grad()

        # --- Recalculate individual losses inside closure for accurate history ---
        T_colloc_unscaled_closure = T_colloc.to(X_colloc.device) # Ensure device
        T_colloc_scaled_input_closure = X_colloc[:, 1:2] # Scaled T

        # Data Loss
        y_pred_data_cl = solution_net(X_data)
        y_pred_data_cl = torch.relu(y_pred_data_cl)
        loss_data_cl = torch.mean((y_pred_data_cl - y_data)**2)

        # Physics Loss
        y_pred_colloc_cl = solution_net(X_colloc)
        y_pred_colloc_cl = torch.relu(y_pred_colloc_cl).clamp(max=1.0)
        dy_dt_scaled_cl = grad(y_pred_colloc_cl, X_colloc,
                               grad_outputs=torch.ones_like(y_pred_colloc_cl),
                               create_graph=True)[0][:, 0:1]
        dy_dt_unscaled_cl = dy_dt_scaled_cl / (1.0 / t_scale_factor)
        A_cl = torch.exp(logA)
        Ea_cl = torch.exp(logEa)
        n_cl = torch.relu(n_net(T_colloc_scaled_input_closure)) # n(T)
        m_cl = torch.relu(m_net(T_colloc_scaled_input_closure)) # m(T)
        K_pred_cl = A_cl * torch.exp(-Ea_cl / (R * T_colloc_unscaled_closure))
        y_clipped_cl = torch.clamp(y_pred_colloc_cl, epsilon, 1.0 - epsilon)
        term1_cl = torch.pow(y_clipped_cl, n_cl)
        term2_cl = torch.pow(1.0 - y_clipped_cl, m_cl)
        f_pred_cl = K_pred_cl * term1_cl * term2_cl
        f_pred_cl = torch.relu(f_pred_cl)
        residual_cl = dy_dt_unscaled_cl - f_pred_cl
        loss_physics_cl = torch.mean(residual_cl**2)

        # Initial Condition Loss
        y_pred_zero_cl = solution_net(X_zero)
        y_pred_zero_cl = torch.relu(y_pred_zero_cl)
        loss_ic_cl = torch.mean(y_pred_zero_cl**2)

        # Combined Loss
        loss = loss_data_cl + lambda_physics * loss_physics_cl + lambda_ic * loss_ic_cl
        # --- End recalculation ---

        loss.backward()

        # Store history
        loss_history.append(loss.item())
        loss_comp_history['data'].append(loss_data_cl.item())
        loss_comp_history['physics'].append(loss_physics_cl.item())
        loss_comp_history['ic'].append(loss_ic_cl.item())
        param_history['A'].append(A_cl.item())
        param_history['Ea'].append(Ea_cl.item())
        # Cannot store single n, m anymore

        return loss

    loss = optimizer.step(closure)
    # scheduler.step()

    if (epoch + 1) % 500 == 0:
        current_loss = loss_history[-1]
        data_l = loss_comp_history['data'][-1]
        phys_l = loss_comp_history['physics'][-1]
        ic_l = loss_comp_history['ic'][-1]
        A_disp = param_history['A'][-1]
        Ea_disp = param_history['Ea'][-1] / 1000.0
        # Cannot display single n, m
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4e}, "
              f"(D:{data_l:.2e}, P:{phys_l:.2e}, IC:{ic_l:.2e}), "
              f"Ea: {Ea_disp:.2f} kJ/mol, A: {A_disp:.2e}, "
              # Removed n, m display
              f"Time: {elapsed_time:.2f}s")
        start_time = time.time()

print("--- Training Finished ---")

final_A = torch.exp(logA).item()
final_Ea = torch.exp(logEa).item()
# Cannot print single final n, m
print("\n--- Final Learned Parameters ---")
print(f"  A : {final_A:.4e} day^-1")
print(f"  Ea: {final_Ea / 1000.0:.3f} kJ/mol")
print(f"  n : Learned via n_net(T)")
print(f"  m : Learned via m_net(T)")
print("-" * 30)


# =============================================================================
# 6. Prediction and Plotting
# =============================================================================
solution_net.eval() # Set networks to evaluation mode
n_net.eval()
m_net.eval()

# --- Plot Loss History ---
# (Loss plotting code remains the same)
plt.figure(figsize=(12, 7))
plt.subplot(2, 1, 1)
plt.semilogy(loss_history, label='Total Loss', linewidth=2)
plt.semilogy(loss_comp_history['data'], label=f'Data Loss (x1)', alpha=0.7)
plt.semilogy([l*lambda_physics for l in loss_comp_history['physics']], label=f'Physics Loss (x{lambda_physics:.1e})', alpha=0.7)
plt.semilogy([l*lambda_ic for l in loss_comp_history['ic']], label=f'IC Loss (x{lambda_ic:.1e})', alpha=0.7)
plt.ylabel("Loss (log scale)")
plt.title("PINN Training Loss History (Autocatalytic, Temp-Dep n,m)")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.subplot(2, 1, 2)
plt.semilogy(loss_comp_history['physics'], label='Physics Loss (Unweighted)', color='red')
plt.ylabel("Physics Loss (log)")
plt.xlabel("Training Iteration")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- Plot Parameter History (A, Ea only) ---
fig_params, axs_params = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs_params[0].plot(param_history['A'])
axs_params[0].set_ylabel('A (day^-1)')
axs_params[0].set_yscale('log')
axs_params[1].plot(np.array(param_history['Ea'])/1000.0) # Plot Ea in kJ/mol
axs_params[1].set_ylabel('Ea (kJ/mol)')
axs_params[1].set_xlabel('Training Iteration')
fig_params.suptitle('Learned Parameter Evolution (A, Ea)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Plot Learned n(T) and m(T) ---
plt.figure(figsize=(12, 5))
temps_plot = np.linspace(T_min_K, T_max_K, 100)
temps_plot_tensor = torch.tensor(temps_plot, dtype=torch.float32).unsqueeze(1).to(device)
temps_plot_scaled = temps_plot_tensor / T_scale_factor
with torch.no_grad():
    n_learned = torch.relu(n_net(temps_plot_scaled)).cpu().numpy()
    m_learned = torch.relu(m_net(temps_plot_scaled)).cpu().numpy()

plt.subplot(1, 2, 1)
plt.plot(temps_plot - 273.15, n_learned)
plt.xlabel("Temperature (°C)")
plt.ylabel("Learned n(T)")
plt.title("Learned n vs Temperature")
plt.grid(True, alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(temps_plot - 273.15, m_learned)
plt.xlabel("Temperature (°C)")
plt.ylabel("Learned m(T)")
plt.title("Learned m vs Temperature")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()


# --- Generate Predictions for Plotting ---
# (Prediction generation code remains the same, uses solution_net)
print("\n--- Generating Predictions for Plotting ---")
prediction_results_list = []
time_pred_vals = np.linspace(0, prediction_days, int(prediction_days / 10) + 1)

with torch.no_grad():
    for _, group in df_all.groupby(['Experiment', 'Temperature_K']):
        experiment_id = group['Experiment'].iloc[0]
        temp_k = group['Temperature_K'].iloc[0]
        temp_c = temp_k - 273.15
        purity_t0 = group['Purity_t0'].iloc[0]

        t_pred_tensor = torch.tensor(time_pred_vals, dtype=torch.float32).unsqueeze(1)
        T_pred_tensor = torch.full_like(t_pred_tensor, temp_k)

        t_pred_scaled = t_pred_tensor / t_scale_factor
        T_pred_scaled = T_pred_tensor / T_scale_factor

        X_pred = torch.cat((t_pred_scaled, T_pred_scaled), dim=1).to(device)

        y_pred_curve = solution_net(X_pred)
        y_pred_curve = torch.relu(y_pred_curve).cpu().numpy().flatten()
        purity_pred_curve = (1.0 - y_pred_curve) * purity_t0

        for t, y_p, p_p in zip(time_pred_vals, y_pred_curve, purity_pred_curve):
             prediction_results_list.append({
                 'Experiment': experiment_id, 'Temperature_C': temp_c, 'Time_days': t,
                 'Predicted_Conversion_y': y_p, 'Predicted_Purity': p_p
             })

df_predictions = pd.DataFrame(prediction_results_list)

# --- Plot Predictions vs Experimental Data ---
# (Plotting code remains the same)
unique_experiments = df_all['Experiment'].unique()
n_exp = len(unique_experiments)
n_cols = 2
n_rows = (n_exp + n_cols - 1) // n_cols
fig_preds, axes_preds = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
axes_preds_flat = axes_preds.flatten()
plot_idx = 0

unique_temps_plot = sorted(df_all['Temperature_C'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps_plot)))
temp_color_map = {temp: color for temp, color in zip(unique_temps_plot, colors)}

for experiment_id in unique_experiments:
    ax = axes_preds_flat[plot_idx]
    exp_data = df_all[df_all['Experiment'] == experiment_id]
    pred_data_exp = df_predictions[df_predictions['Experiment'] == experiment_id]

    for temp_c in sorted(exp_data['Temperature_C'].unique()):
        color = temp_color_map.get(temp_c, 'grey')

        exp_subset = exp_data[exp_data['Temperature_C'] == temp_c]
        ax.scatter(exp_subset['Time_days'], exp_subset['Purity'],
                   label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)

        pred_subset = pred_data_exp[pred_data_exp['Temperature_C'] == temp_c]
        ax.plot(pred_subset['Time_days'], pred_subset['Predicted_Purity'],
                '--', label=f'{temp_c:.0f}°C (PINN)', color=color)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Purity (%)")
    ax.set_title(f"Experiment: {experiment_id} - PINN Predictions (Temp-Dep n,m)")
    ax.axvline(x=max_training_days, color='grey', linestyle=':', label=f'Train Cutoff')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    min_purity_exp = exp_data['Purity'].min() if not exp_data.empty else 0
    max_purity_exp = exp_data['Purity_t0'].max() if not exp_data.empty else 100
    ax.set_ylim(bottom=max(0, min_purity_exp - 5), top=max_purity_exp + 2)
    ax.set_xlim(left=-prediction_days*0.02, right=prediction_days*1.05)
    plot_idx += 1

for i in range(plot_idx, len(axes_preds_flat)): fig_preds.delaxes(axes_preds_flat[i])
fig_preds.suptitle(f"PINN Model Predictions (Temp-Dep n,m, Extrapolated to {prediction_days} days, lambda_phys={lambda_physics:.1e})", fontsize=16, y=1.02)
fig_preds.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# =============================================================================
# 7. Quantitative Evaluation Metrics
# =============================================================================
# (Evaluation metrics code remains the same)
print("\n--- Calculating Evaluation Metrics ---")
df_eval = df_all.copy()
t_eval_exp = torch.tensor(df_eval['Time_days'].values, dtype=torch.float32).unsqueeze(1)
T_eval_exp = torch.tensor(df_eval['Temperature_K'].values, dtype=torch.float32).unsqueeze(1)
t_eval_scaled = t_eval_exp / t_scale_factor
T_eval_scaled = T_eval_exp / T_scale_factor
X_eval = torch.cat((t_eval_scaled, T_eval_scaled), dim=1).to(device)
with torch.no_grad():
    y_pred_eval = solution_net(X_eval)
    y_pred_eval = torch.relu(y_pred_eval).cpu().numpy().flatten()
df_eval['Predicted_Conversion_y'] = y_pred_eval
df_eval['Predicted_Purity'] = (1.0 - df_eval['Predicted_Conversion_y']) * df_eval['Purity_t0']
df_eval['Predicted_Purity'] = df_eval['Predicted_Purity'].clip(lower=0)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    if np.sum(non_zero_idx) == 0: return np.nan
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / (y_true[non_zero_idx] + epsilon))) * 100
eval_train = df_eval[df_eval['Time_days'] <= max_training_days]
eval_valid = df_eval[df_eval['Time_days'] > max_training_days]
metrics_results = {}
print("--- Model Evaluation Metrics (Based on Purity Predictions vs Experimental) ---")
for label, df_subset in [('Training', eval_train), ('Validation', eval_valid), ('Overall', df_eval)]:
    df_subset_no_t0 = df_subset[df_subset['Time_days'] > 0]
    if len(df_subset_no_t0) > 0:
        y_true = df_subset_no_t0['Purity']
        y_pred = df_subset_no_t0['Predicted_Purity']
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_results[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_subset_no_t0)}
        print(f"\nMetrics for {label} Data (N={len(df_subset_no_t0)}):")
        print(f"  Mean Absolute Error (MAE):     {mae:.3f} % Purity")
        print(f"  Root Mean Squared Error (RMSE):{rmse:.3f} % Purity")
        print(f"  R-squared (R2):                {r2:.4f}")
        print(f"  Mean Abs Percentage Error (MAPE):{mape:.2f} %")
    else:
        print(f"\nNo data available for {label} set evaluation (excluding t=0).")
print("-" * 40)


# =============================================================================
# 8. Save Extrapolated Predictions to Excel
# =============================================================================
if not df_predictions.empty:
    try:
        # Add learned parameters to the prediction dataframe for reference
        df_predictions['A_learned'] = final_A
        df_predictions['Ea_learned_kJmol'] = final_Ea / 1000.0
        df_predictions['n_model'] = 'NN(T)' # Indicate n is temp-dependent
        df_predictions['m_model'] = 'NN(T)' # Indicate m is temp-dependent
        df_predictions['lambda_physics'] = lambda_physics
        df_predictions['lambda_ic'] = lambda_ic
        df_predictions['epochs'] = num_epochs

        df_predictions_sorted = df_predictions.sort_values(by=['Experiment', 'Temperature_C', 'Time_days'])
        df_predictions_sorted.to_excel(predictions_filename, index=False, engine='openpyxl')
        print(f"\n--- Extrapolated predictions saved to '{predictions_filename}' ---")
    except Exception as e:
        print(f"\nERROR: Could not save predictions to Excel file '{predictions_filename}': {e}")
else:
    print("\nNo extrapolated predictions were generated to save.")
print("-" * 40)


# =============================================================================
# 9. Simulate Learned Parameters with solve_ivp (Updated for n(T), m(T))
# =============================================================================
print("\n--- Simulating Dynamics using Final Learned Parameters & solve_ivp ---")

# Need n_net and m_net in the ODE function scope, or pass them
# Let's redefine the ODE function here to capture them
def reaction_ode_check_temp_nm(t, y, temp_k_check, A_check, Ea_check, n_net_check, m_net_check, T_scale_factor_check, device_check):
    """ ODE function using n(T) and m(T) from networks. """
    y_scalar = y[0]
    y_safe = np.clip(y_scalar, epsilon, 1.0 - epsilon)

    # Calculate n and m for the current temperature
    T_k_tensor = torch.tensor([[temp_k_check]], dtype=torch.float32).to(device_check)
    T_k_scaled = T_k_tensor / T_scale_factor_check
    with torch.no_grad():
        n_check = torch.relu(n_net_check(T_k_scaled)).item()
        m_check = torch.relu(m_net_check(T_k_scaled)).item()

    # Calculate K(T)
    K_check = A_check * np.exp(-Ea_check / (R * temp_k_check))

    # Calculate rate
    # Use np.power for potentially non-integer exponents from networks
    term1 = np.power(y_safe, n_check)
    term2 = np.power(1.0 - y_safe, m_check)
    dydt = K_check * term1 * term2
    return [max(0.0, dydt)]

plt.figure(figsize=(10, 6))
plt.title("Simulation using Final Learned Parameters (Temp-Dep n,m) & solve_ivp")
time_solve_ivp = np.linspace(0, prediction_days, 200)

solve_ivp_results = {}

# Ensure networks are in eval mode for solve_ivp check
n_net.eval()
m_net.eval()

for temp_c in sorted(df_all['Temperature_C'].unique()):
    temp_k_check = temp_c + 273.15
    purity_t0_check = df_all[(df_all['Temperature_C'] == temp_c) & (df_all['Time_days'] == 0)]['Purity'].mean()
    if pd.isna(purity_t0_check):
         purity_t0_check = df_all[df_all['Temperature_C'] == temp_c]['Purity_t0'].iloc[0]

    color = temp_color_map.get(temp_c, 'grey')

    y0_check = [0.0] # Initial conversion
    t_span_check = [0, prediction_days]

    try:
        # Pass necessary args: temp_k, A, Ea, networks, scaling factor, device
        sol = solve_ivp(reaction_ode_check_temp_nm, t_span_check, y0_check,
                        args=(temp_k_check, final_A, final_Ea, n_net, m_net, T_scale_factor, device),
                        t_eval=time_solve_ivp, method='RK45', rtol=1e-6, atol=1e-8)

        if sol.status == 0:
            y_solved = sol.y[0]
            purity_solved = (1.0 - y_solved) * purity_t0_check
            plt.plot(sol.t, purity_solved, '-', label=f'{temp_c:.0f}°C (solve_ivp)', color=color)
            solve_ivp_results[temp_c] = pd.DataFrame({'Time_days': sol.t, 'Purity_solve_ivp': purity_solved})
        else:
            print(f"solve_ivp failed for {temp_c}°C with status {sol.status}: {sol.message}")
            plt.plot([], [], '-', label=f'{temp_c:.0f}°C (solve_ivp failed)', color=color)

    except Exception as e:
         print(f"Exception during solve_ivp for {temp_c}°C: {e}")
         # Consider adding traceback print here for debugging
         # import traceback
         # traceback.print_exc()
         plt.plot([], [], '-', label=f'{temp_c:.0f}°C (solve_ivp error)', color=color)


plt.xlabel("Time (days)")
plt.ylabel("Purity (%)")
plt.legend(fontsize='small')
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(bottom=-5)
plt.show()
print("-" * 40)


print("Script finished.")
