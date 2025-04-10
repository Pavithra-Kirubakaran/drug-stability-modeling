# -*- coding: utf-8 -*-
"""
Conceptual Python script for predicting drug stability using a
Physics-Informed Neural Network (PINN) with PyTorch.

*** Version 7: Added Quantitative Metrics for Final solve_ivp Predictions ***
*** Uses PINN to find A, Ea, n(T), m(T) ***
*** Uses solve_ivp with learned components for final prediction ***
*** Set lambda_physics = 1.0 ***

This script outlines the structure for:
1. Loading and preparing data.
2. Defining Neural Networks (NNs):
   - solution_net: Approximates conversion y(t, T) (used during training).
   - n_net: Predicts reaction order n as a function of T.
   - m_net: Predicts reaction order m as a function of T.
3. Defining learnable parameters for Arrhenius kinetics:
   - logA, logEa (for K = A * exp(-Ea / RT))
4. Defining the PINN loss function to guide parameter/network learning.
5. Setting up the training loop using PyTorch's optimizer.
6. Saving learned parameters (A, Ea) and network weights (n_net, m_net).
7. Evaluating the fit of solution_net (primarily for diagnostics).
8. Using solve_ivp with the learned A, Ea, n_net, m_net for final prediction,
   plotting, and saving.
9. Calculating quantitative metrics (MAE, RMSE, R2, MAPE) for the
   final solve_ivp predictions against experimental data.


NOTE: The primary output is now from solve_ivp (Section 9).
      solution_net's direct predictions are secondary/diagnostic.
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
import os # For saving files

# Suppress potential warnings from metrics calculation if needed
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Configuration & Setup
# =============================================================================
excel_filename = 'Data/stability_data.xlsx' # Input data file
# Output filenames
predictions_filename_solve_ivp = 'purity_predictions_solve_ivp_final.xlsx'
n_net_weights_filename = 'n_net_weights.pth'
m_net_weights_filename = 'm_net_weights.pth'

# PINN Hyperparameters (EXAMPLES - NEED TUNING)
learning_rate = 1e-3
num_epochs = 20000
# Set lambda_physics for parameter discovery phase
lambda_physics = 1.0 # Weight for the physics loss component
lambda_ic = 1.0 # Weight for the initial condition loss
nn_arch_solution = [2, 50, 50, 50, 1] # Input (t_scaled, T_scaled), Output (y)
# Architectures for temperature-dependent n and m networks
nn_arch_n = [1, 10, 10, 1] # Input (T_scaled), Hidden layers, Output (n)
nn_arch_m = [1, 10, 10, 1] # Input (T_scaled), Hidden layers, Output (m)


max_training_days = 90 # Use data up to this time for training loss
prediction_days = 3 * 365 # Extrapolation time
num_collocation_points = 20000 # Number of points to enforce physics loss

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
    df_all = pd.read_excel(excel_filename,'Sheet1')
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
        self.activation = nn.SiLU()
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
initial_logA = np.log(1e10)
initial_logEa = np.log(83140.0)

logA = nn.Parameter(torch.tensor(initial_logA, dtype=torch.float32, device=device))
logEa = nn.Parameter(torch.tensor(initial_logEa, dtype=torch.float32, device=device))


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
    y_pred_data = torch.relu(y_pred_data)
    loss_data = torch.mean((y_pred_data - y_data)**2)

    # --- Physics Loss ---
    T_colloc_scaled_input = X_colloc[:, 1:2]
    y_pred_colloc = solution_net(X_colloc)
    y_pred_colloc = torch.relu(y_pred_colloc).clamp(max=1.0)

    dy_dt_scaled = grad(y_pred_colloc, X_colloc,
                        grad_outputs=torch.ones_like(y_pred_colloc),
                        create_graph=True)[0][:, 0:1]
    dy_dt_unscaled = dy_dt_scaled / (1.0 / t_scale_factor)

    A = torch.exp(logA)
    Ea = torch.exp(logEa)
    # Ensure networks are in training mode if needed (should be handled by main loop)
    n = torch.relu(n_net(T_colloc_scaled_input))
    m = torch.relu(m_net(T_colloc_scaled_input))
    K_pred = A * torch.exp(-Ea / (R * T_colloc_unscaled))

    y_clipped = torch.clamp(y_pred_colloc, epsilon, 1.0 - epsilon)
    term1 = torch.pow(y_clipped, n)
    term2 = torch.pow(1.0 - y_clipped, m)
    f_pred = K_pred * term1 * term2
    f_pred = torch.relu(f_pred)

    residual = dy_dt_unscaled - f_pred
    loss_physics = torch.mean(residual**2)

    # --- Initial Condition Loss (y(0, T) = 0) ---
    y_pred_zero = solution_net(X_zero)
    y_pred_zero = torch.relu(y_pred_zero)
    loss_ic = torch.mean(y_pred_zero**2)

    # --- Combined Loss ---
    total_loss = loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic

    return total_loss, A, Ea

# =============================================================================
# 4. Training Setup
# =============================================================================
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
print("\n--- Starting PINN Training (Parameter Discovery Phase) ---")
start_time = time.time()
loss_history = []
param_history = {'A': [], 'Ea': []}
loss_comp_history = {'data': [], 'physics': [], 'ic': []}

for epoch in range(num_epochs):
    solution_net.train()
    n_net.train()
    m_net.train()

    def closure():
        optimizer.zero_grad()
        # --- Recalculate individual losses inside closure for accurate history ---
        T_colloc_unscaled_closure = T_colloc.to(X_colloc.device)
        T_colloc_scaled_input_closure = X_colloc[:, 1:2]
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
        # Ensure networks are in training mode for loss calculation if dropout/batchnorm were used
        # (Not strictly necessary here, but good practice)
        n_cl = torch.relu(n_net(T_colloc_scaled_input_closure))
        m_cl = torch.relu(m_net(T_colloc_scaled_input_closure))
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
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4e}, "
              f"(D:{data_l:.2e}, P:{phys_l:.2e}, IC:{ic_l:.2e}), "
              f"Ea: {Ea_disp:.2f} kJ/mol, A: {A_disp:.2e}, "
              f"Time: {elapsed_time:.2f}s")
        start_time = time.time()

print("--- Training Finished ---")

final_A = torch.exp(logA).item()
final_Ea = torch.exp(logEa).item()
print("\n--- Final Learned Parameters ---")
print(f"  A : {final_A:.4e} day^-1")
print(f"  Ea: {final_Ea / 1000.0:.3f} kJ/mol")
print(f"  n : Learned via n_net(T)")
print(f"  m : Learned via m_net(T)")
print("-" * 30)

# =============================================================================
# 6. Save Learned Parameters and Network Weights (Added)
# =============================================================================
print("\n--- Saving Learned Components ---")
try:
    # Save n_net and m_net state dictionaries
    torch.save(n_net.state_dict(), n_net_weights_filename)
    torch.save(m_net.state_dict(), m_net_weights_filename)
    print(f"Saved n_net weights to {n_net_weights_filename}")
    print(f"Saved m_net weights to {m_net_weights_filename}")
    # Parameters A and Ea are available as final_A, final_Ea
except Exception as e:
    print(f"ERROR saving network weights: {e}")
print("-" * 40)


# =============================================================================
# 7. Diagnostic Evaluation & Plotting of solution_net
#    (This section is now secondary, mainly for checking PINN training)
# =============================================================================
solution_net.eval()
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
plt.title("PINN Training Loss History (Parameter Discovery Phase)")
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
# (Plotting code remains the same)
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
# (Plotting code remains the same)
plt.figure(figsize=(12, 5))
temps_plot = np.linspace(T_min_K, T_max_K, 100)
temps_plot_tensor = torch.tensor(temps_plot, dtype=torch.float32).unsqueeze(1).to(device)
temps_plot_scaled = temps_plot_tensor / T_scale_factor
with torch.no_grad():
    n_learned_func = torch.relu(n_net(temps_plot_scaled)).cpu().numpy()
    m_learned_func = torch.relu(m_net(temps_plot_scaled)).cpu().numpy()

plt.subplot(1, 2, 1)
plt.plot(temps_plot - 273.15, n_learned_func)
plt.xlabel("Temperature (°C)")
plt.ylabel("Learned n(T)")
plt.title("Learned n vs Temperature")
plt.grid(True, alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(temps_plot - 273.15, m_learned_func)
plt.xlabel("Temperature (°C)")
plt.ylabel("Learned m(T)")
plt.title("Learned m vs Temperature")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()


# --- Generate Predictions using solution_net (for comparison/diagnostics only) ---
print("\n--- Generating solution_net Predictions (Diagnostic) ---")
solution_net_predictions_list = []
time_pred_vals_diag = np.linspace(0, prediction_days, int(prediction_days / 10) + 1)

with torch.no_grad():
    for _, group in df_all.groupby(['Experiment', 'Temperature_K']):
        # ... (prediction code identical to previous version, store in solution_net_predictions_list) ...
        experiment_id = group['Experiment'].iloc[0]
        temp_k = group['Temperature_K'].iloc[0]
        temp_c = temp_k - 273.15
        purity_t0 = group['Purity_t0'].iloc[0]

        t_pred_tensor = torch.tensor(time_pred_vals_diag, dtype=torch.float32).unsqueeze(1)
        T_pred_tensor = torch.full_like(t_pred_tensor, temp_k)
        t_pred_scaled = t_pred_tensor / t_scale_factor
        T_pred_scaled = T_pred_tensor / T_scale_factor
        X_pred = torch.cat((t_pred_scaled, T_pred_scaled), dim=1).to(device)

        y_pred_curve = solution_net(X_pred)
        y_pred_curve = torch.relu(y_pred_curve).cpu().numpy().flatten()
        purity_pred_curve = (1.0 - y_pred_curve) * purity_t0

        for t, y_p, p_p in zip(time_pred_vals_diag, y_pred_curve, purity_pred_curve):
             solution_net_predictions_list.append({
                 'Experiment': experiment_id, 'Temperature_C': temp_c, 'Time_days': t,
                 'Predicted_Conversion_y': y_p, 'Predicted_Purity': p_p
             })

df_predictions_solution_net = pd.DataFrame(solution_net_predictions_list)


# --- Plot solution_net Predictions vs Experimental Data (Diagnostic) ---
# (Plotting code identical, but uses df_predictions_solution_net and different title)
unique_experiments = df_all['Experiment'].unique()
n_exp = len(unique_experiments)
n_cols = 2
n_rows = (n_exp + n_cols - 1) // n_cols
fig_preds_solnet, axes_preds_solnet = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
axes_preds_flat_solnet = axes_preds_solnet.flatten()
plot_idx = 0
unique_temps_plot = sorted(df_all['Temperature_C'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps_plot)))
temp_color_map = {temp: color for temp, color in zip(unique_temps_plot, colors)}
for experiment_id in unique_experiments:
    ax = axes_preds_flat_solnet[plot_idx]
    exp_data = df_all[df_all['Experiment'] == experiment_id]
    pred_data_exp = df_predictions_solution_net[df_predictions_solution_net['Experiment'] == experiment_id] # Use solution_net results
    for temp_c in sorted(exp_data['Temperature_C'].unique()):
        color = temp_color_map.get(temp_c, 'grey')
        exp_subset = exp_data[exp_data['Temperature_C'] == temp_c]
        ax.scatter(exp_subset['Time_days'], exp_subset['Purity'],
                   label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)
        pred_subset = pred_data_exp[pred_data_exp['Temperature_C'] == temp_c]
        if not pred_subset.empty: # Add check if prediction exists
            ax.plot(pred_subset['Time_days'], pred_subset['Predicted_Purity'],
                    '--', label=f'{temp_c:.0f}°C (solution_net)', color=color) # Label change
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Purity (%)")
    ax.set_title(f"Experiment: {experiment_id} - solution_net Predictions (Diagnostic)") # Title change
    ax.axvline(x=max_training_days, color='grey', linestyle=':', label=f'Train Cutoff')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    min_purity_exp = exp_data['Purity'].min() if not exp_data.empty else 0
    max_purity_exp = exp_data['Purity_t0'].max() if not exp_data.empty else 100
    ax.set_ylim(bottom=max(0, min_purity_exp - 5), top=max_purity_exp + 2)
    ax.set_xlim(left=-prediction_days*0.02, right=prediction_days*1.05)
    plot_idx += 1
for i in range(plot_idx, len(axes_preds_flat_solnet)): fig_preds_solnet.delaxes(axes_preds_flat_solnet[i])
fig_preds_solnet.suptitle(f"PINN solution_net Predictions (Diagnostic, lambda_phys={lambda_physics:.1e})", fontsize=16, y=1.02) # Title change
fig_preds_solnet.tight_layout(rect=[0, 0, 1, 1])
plt.show()


# =============================================================================
# 8. Quantitative Evaluation Metrics (Based on solution_net - Diagnostic)
# =============================================================================
# (Evaluation metrics code remains the same, reports on solution_net performance)
print("\n--- Calculating Evaluation Metrics (for solution_net - Diagnostic) ---")
df_eval_solnet = df_all.copy() # Use a distinct name
t_eval_exp_solnet = torch.tensor(df_eval_solnet['Time_days'].values, dtype=torch.float32).unsqueeze(1)
T_eval_exp_solnet = torch.tensor(df_eval_solnet['Temperature_K'].values, dtype=torch.float32).unsqueeze(1)
t_eval_scaled_solnet = t_eval_exp_solnet / t_scale_factor
T_eval_scaled_solnet = T_eval_exp_solnet / T_scale_factor
X_eval_solnet = torch.cat((t_eval_scaled_solnet, T_eval_scaled_solnet), dim=1).to(device)
with torch.no_grad():
    y_pred_eval_solnet = solution_net(X_eval_solnet)
    y_pred_eval_solnet = torch.relu(y_pred_eval_solnet).cpu().numpy().flatten()
df_eval_solnet['Predicted_Conversion_y'] = y_pred_eval_solnet
df_eval_solnet['Predicted_Purity'] = (1.0 - df_eval_solnet['Predicted_Conversion_y']) * df_eval_solnet['Purity_t0']
df_eval_solnet['Predicted_Purity'] = df_eval_solnet['Predicted_Purity'].clip(lower=0)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    if np.sum(non_zero_idx) == 0: return np.nan
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / (y_true[non_zero_idx] + epsilon))) * 100
eval_train_solnet = df_eval_solnet[df_eval_solnet['Time_days'] <= max_training_days]
eval_valid_solnet = df_eval_solnet[df_eval_solnet['Time_days'] > max_training_days]
metrics_results_solnet = {}
print("--- solution_net Evaluation Metrics (Diagnostic) ---")
for label, df_subset in [('Training', eval_train_solnet), ('Validation', eval_valid_solnet), ('Overall', df_eval_solnet)]:
    df_subset_no_t0 = df_subset[df_subset['Time_days'] > 0]
    if len(df_subset_no_t0) > 0:
        y_true = df_subset_no_t0['Purity']
        y_pred = df_subset_no_t0['Predicted_Purity']
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_results_solnet[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_subset_no_t0)}
        print(f"\nMetrics for {label} Data (N={len(df_subset_no_t0)}):")
        print(f"  Mean Absolute Error (MAE):     {mae:.3f} % Purity")
        print(f"  Root Mean Squared Error (RMSE):{rmse:.3f} % Purity")
        print(f"  R-squared (R2):                {r2:.4f}")
        print(f"  Mean Abs Percentage Error (MAPE):{mape:.2f} %")
    else:
        print(f"\nNo data available for {label} set evaluation (excluding t=0).")
print("-" * 40)


# =============================================================================
# 9. Generate Final Predictions using solve_ivp + Learned Components
#    (This section is now the primary prediction output)
# =============================================================================
print("\n--- Generating Final Predictions using solve_ivp + Learned Components ---")

def reaction_ode_final(t, y, temp_k_check, A_check, Ea_check, n_net_check, m_net_check, T_scale_factor_check, device_check):
    """ ODE function using n(T) and m(T) from networks. """
    y_scalar = y[0]
    y_safe = np.clip(y_scalar, epsilon, 1.0 - epsilon)
    # Calculate n and m for the current temperature
    T_k_tensor = torch.tensor([[temp_k_check]], dtype=torch.float32).to(device_check)
    T_k_scaled = T_k_tensor / T_scale_factor_check
    with torch.no_grad():
        # Ensure networks are in eval mode
        n_net_check.eval()
        m_net_check.eval()
        n_check = torch.relu(n_net_check(T_k_scaled)).item()
        m_check = torch.relu(m_net_check(T_k_scaled)).item()
    # Calculate K(T)
    K_check = A_check * np.exp(-Ea_check / (R * temp_k_check))
    # Calculate rate
    term1 = np.power(y_safe, n_check)
    term2 = np.power(1.0 - y_safe, m_check)
    dydt = K_check * term1 * term2
    return [max(0.0, dydt)]

# Use finer time steps for solve_ivp final predictions
time_solve_ivp_final = np.linspace(0, prediction_days, int(prediction_days / 2) + 1) # More points
solve_ivp_final_results_list = []

# Ensure networks are in eval mode before loop
n_net.eval()
m_net.eval()

for temp_c in sorted(df_all['Temperature_C'].unique()):
    temp_k_check = temp_c + 273.15
    # Find corresponding Purity_t0 - handle multiple experiments if necessary
    # Use the first Purity_t0 found for this temperature
    purity_t0_check = df_all[df_all['Temperature_C'] == temp_c]['Purity_t0'].iloc[0]
    # Get all experiments for this temperature to generate predictions for each
    experiments_at_temp = df_all[df_all['Temperature_C'] == temp_c]['Experiment'].unique()

    color = temp_color_map.get(temp_c, 'grey')
    y0_check = [0.0] # Initial conversion
    t_span_check = [0, prediction_days]

    try:
        # Pass necessary args: temp_k, A, Ea, networks, scaling factor, device
        sol = solve_ivp(reaction_ode_final, t_span_check, y0_check,
                        args=(temp_k_check, final_A, final_Ea, n_net, m_net, T_scale_factor, device),
                        t_eval=time_solve_ivp_final, method='RK45', rtol=1e-6, atol=1e-8)

        if sol.status == 0:
            y_solved = sol.y[0]
            purity_solved = (1.0 - y_solved) * purity_t0_check
            # Store results for this temperature for EACH experiment
            for exp_id in experiments_at_temp:
                for t, y_sol, p_sol in zip(sol.t, y_solved, purity_solved):
                     solve_ivp_final_results_list.append({
                         'Experiment': exp_id, # Include experiment ID
                         'Temperature_C': temp_c,
                         'Time_days': t,
                         'Predicted_Conversion_y': y_sol,
                         'Predicted_Purity': p_sol
                     })
        else:
            print(f"solve_ivp failed for final prediction at {temp_c}°C with status {sol.status}: {sol.message}")

    except Exception as e:
         print(f"Exception during final solve_ivp for {temp_c}°C: {e}")

# Create DataFrame from solve_ivp results
df_predictions_final = pd.DataFrame(solve_ivp_final_results_list)

# --- Plot Final Predictions (solve_ivp) vs Experimental Data ---
fig_preds_final, axes_preds_final = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
axes_preds_flat_final = axes_preds_final.flatten()
plot_idx = 0
for experiment_id in unique_experiments:
    ax = axes_preds_flat_final[plot_idx]
    exp_data = df_all[df_all['Experiment'] == experiment_id]
    # Filter final predictions for this experiment
    pred_data_exp_final = df_predictions_final[df_predictions_final['Experiment'] == experiment_id]

    for temp_c in sorted(exp_data['Temperature_C'].unique()):
        color = temp_color_map.get(temp_c, 'grey')
        # Plot experimental data
        exp_subset = exp_data[exp_data['Temperature_C'] == temp_c]
        ax.scatter(exp_subset['Time_days'], exp_subset['Purity'],
                   label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)
        # Plot final prediction curve from solve_ivp
        pred_subset_final = pred_data_exp_final[pred_data_exp_final['Temperature_C'] == temp_c]
        if not pred_subset_final.empty:
            ax.plot(pred_subset_final['Time_days'], pred_subset_final['Predicted_Purity'],
                    '-', label=f'{temp_c:.0f}°C (Final Pred)', color=color) # Solid line for final prediction

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Purity (%)")
    ax.set_title(f"Experiment: {experiment_id} - Final Predictions (solve_ivp)")
    ax.axvline(x=max_training_days, color='grey', linestyle=':', label=f'Train Cutoff')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    min_purity_exp = exp_data['Purity'].min() if not exp_data.empty else 0
    max_purity_exp = exp_data['Purity_t0'].max() if not exp_data.empty else 100
    ax.set_ylim(bottom=max(0, min_purity_exp - 5), top=max_purity_exp + 2)
    ax.set_xlim(left=-prediction_days*0.02, right=prediction_days*1.05)
    plot_idx += 1
for i in range(plot_idx, len(axes_preds_flat_final)): fig_preds_final.delaxes(axes_preds_flat_final[i])
fig_preds_final.suptitle(f"Final Predictions using solve_ivp + Learned Components", fontsize=16, y=1.02)
fig_preds_final.tight_layout(rect=[0, 0, 1, 1])
plt.show()


# =============================================================================
# 10. Save Final Predictions (solve_ivp) to Excel (Updated)
# =============================================================================
if not df_predictions_final.empty:
    try:
        # Add learned parameters to the prediction dataframe for reference
        df_predictions_final['A_learned'] = final_A
        df_predictions_final['Ea_learned_kJmol'] = final_Ea / 1000.0
        df_predictions_final['n_model'] = 'NN(T)'
        df_predictions_final['m_model'] = 'NN(T)'
        df_predictions_final['lambda_physics'] = lambda_physics
        df_predictions_final['lambda_ic'] = lambda_ic
        df_predictions_final['epochs'] = num_epochs

        df_predictions_final_sorted = df_predictions_final.sort_values(by=['Experiment', 'Temperature_C', 'Time_days'])
        df_predictions_final_sorted.to_excel(predictions_filename_solve_ivp, index=False, engine='openpyxl')
        print(f"\n--- Final solve_ivp predictions saved to '{predictions_filename_solve_ivp}' ---")
    except Exception as e:
        print(f"\nERROR: Could not save final predictions to Excel file '{predictions_filename_solve_ivp}': {e}")
else:
    print("\nNo final solve_ivp predictions were generated to save.")
print("-" * 40)

# =============================================================================
# 11. Quantitative Evaluation Metrics for Final solve_ivp Predictions (Added)
# =============================================================================
print("\n--- Calculating Evaluation Metrics for Final solve_ivp Predictions ---")
df_eval_final = df_all.copy() # Start with all experimental data
df_eval_final['Predicted_Purity_Final'] = np.nan # Add column for predictions

# Ensure networks are in eval mode
n_net.eval()
m_net.eval()

# Loop through conditions and get predictions AT experimental time points
unique_conditions = df_eval_final[['Experiment', 'Temperature_K']].drop_duplicates()
for index, row in unique_conditions.iterrows():
    exp_id = row['Experiment']
    temp_k_eval = row['Temperature_K']

    # Get experimental time points for this specific condition
    condition_mask = (df_eval_final['Experiment'] == exp_id) & (df_eval_final['Temperature_K'] == temp_k_eval)
    t_eval_points = df_eval_final.loc[condition_mask, 'Time_days'].unique()
    t_eval_points = np.sort(t_eval_points[t_eval_points >= 0]) # Ensure sorted, non-negative

    if len(t_eval_points) == 0: continue

    # Get Purity_t0 for this condition
    purity_t0_eval = df_eval_final.loc[condition_mask, 'Purity_t0'].iloc[0]

    y0_eval = [0.0] # Initial conversion
    t_span_eval = [0, max(t_eval_points) * 1.01] # Span slightly beyond last point

    try:
        sol_eval = solve_ivp(reaction_ode_final, t_span_eval, y0_eval,
                             args=(temp_k_eval, final_A, final_Ea, n_net, m_net, T_scale_factor, device),
                             t_eval=t_eval_points, # Evaluate ONLY at experimental times
                             method='RK45', rtol=1e-6, atol=1e-8)

        if sol_eval.status == 0:
            y_solved_eval = sol_eval.y[0]
            purity_solved_eval = (1.0 - y_solved_eval) * purity_t0_eval
            # Create a mapping from time to predicted purity
            time_to_purity_map = dict(zip(sol_eval.t, purity_solved_eval))
            # Fill the prediction column using the map
            df_eval_final.loc[condition_mask, 'Predicted_Purity_Final'] = df_eval_final.loc[condition_mask, 'Time_days'].map(time_to_purity_map)
        else:
            print(f"solve_ivp failed for metrics calculation at {temp_k_eval-273.15:.1f}°C, Exp: {exp_id}")

    except Exception as e:
        print(f"Exception during solve_ivp for metrics calculation at {temp_k_eval-273.15:.1f}°C, Exp: {exp_id}: {e}")

# Drop rows where prediction failed
df_eval_final = df_eval_final.dropna(subset=['Predicted_Purity_Final', 'Purity'])

# Calculate metrics
eval_train_final = df_eval_final[df_eval_final['Time_days'] <= max_training_days]
eval_valid_final = df_eval_final[df_eval_final['Time_days'] > max_training_days]
metrics_results_final = {}
print("--- Final Model Evaluation Metrics (solve_ivp Predictions vs Experimental) ---")
for label, df_subset in [('Training', eval_train_final), ('Validation', eval_valid_final), ('Overall', df_eval_final)]:
    df_subset_no_t0 = df_subset[df_subset['Time_days'] > 0]
    if len(df_subset_no_t0) > 0:
        y_true = df_subset_no_t0['Purity']
        y_pred = df_subset_no_t0['Predicted_Purity_Final'] # Use the final predictions
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_results_final[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_subset_no_t0)}
        print(f"\nMetrics for {label} Data (N={len(df_subset_no_t0)}):")
        print(f"  Mean Absolute Error (MAE):     {mae:.3f} % Purity")
        print(f"  Root Mean Squared Error (RMSE):{rmse:.3f} % Purity")
        print(f"  R-squared (R2):                {r2:.4f}")
        print(f"  Mean Abs Percentage Error (MAPE):{mape:.2f} %")
    else:
        print(f"\nNo data available for {label} set evaluation (excluding t=0).")
print("-" * 40)


print("Script finished.")
