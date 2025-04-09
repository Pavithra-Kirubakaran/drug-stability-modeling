# -*- coding: utf-8 -*-
"""
Conceptual Implementation of a Neural ODE approach for modeling
purity degradation using PyTorch and torchdiffeq.

NOTE: This code requires PyTorch and torchdiffeq.
      It cannot be run in this environment.
      It serves as a structural example.

Problem: Learn the degradation dynamics dy/dt = f(y, T; theta) directly
         from experimental data, where y is conversion.

Model: A Neural Network (NN_ode) approximates the function f.
       An ODE solver integrates dy/dt = NN_ode(y, T) to predict y(t).

Changes:
- Fixed RuntimeError: Mismatch in shape by ensuring consistent tensor shapes
  [batch, state_dim=1] for y0, y, and dy/dt within ODEFunc and odeint call.
- Maintained previous fixes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# --- PyTorch and torchdiffeq Imports ---

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
torch_available = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# Configuration & Constants << --- USER TUNING AREA --- >>
# =============================================================================
excel_filename = 'purity_data.xlsx'
predictions_filename = 'purity_predictions_NeuralODE_3yr.xlsx'
EXCLUDE_TEMPS_C_TRAINING = [5.0]
N_ODE_HIDDEN_LAYERS = 3
N_ODE_NEURONS = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 10000
ODE_SOLVER_METHOD = 'dopri5'
ODE_RTOL = 1e-4
ODE_ATOL = 1e-5
max_training_days = 90
R_GAS = 8.314
prediction_days = 3 * 365

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================
# ... (load_and_prepare_data function is the same) ...
def load_and_prepare_data(filename, exclude_temps_C=None):
    if exclude_temps_C is None: exclude_temps_C = []
    try: df = pd.read_excel(filename)
    except FileNotFoundError: print(f"ERROR: Excel file '{filename}' not found."); return None, None, None, None
    df = df.dropna(subset=['Experiment', 'Temperature_C', 'Time_days', 'Purity'])
    df = df.astype({'Temperature_C': float, 'Time_days': float, 'Purity': float})
    df = df[df['Purity'] >= 0];
    if df.empty: print("ERROR: No valid data."); return None, None, None, None
    initial_rows = len(df); df = df[~df['Temperature_C'].isin(exclude_temps_C)]
    if len(df) < initial_rows: print(f"Excluded {initial_rows - len(df)} rows for temperatures: {exclude_temps_C}")
    if df.empty: print(f"ERROR: No data remaining after excluding temperatures."); return None, None, None, None
    df['Temperature_K'] = df['Temperature_C'] + 273.15; df['Time_rank'] = df.groupby(['Experiment', 'Temperature_C'])['Time_days'].rank(method='first')
    purity_t0_map = df.loc[df['Time_rank'] == 1].set_index(['Experiment', 'Temperature_C'])['Purity']
    df['Purity_t0'] = df.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map)
    df['Purity_t0'] = df['Purity_t0'].fillna(method='ffill').fillna(method='bfill').replace(0, np.nan); df['Fraction_Remaining'] = np.nan
    valid_p0_idx = df['Purity_t0'].notna() & (df['Purity_t0'] > 0)
    df.loc[valid_p0_idx, 'Fraction_Remaining'] = (df.loc[valid_p0_idx, 'Purity'] / df.loc[valid_p0_idx, 'Purity_t0'])
    df['Fraction_Remaining'] = df['Fraction_Remaining'].fillna(1.0).clip(upper=1.0)
    df['Conversion_y'] = (1.0 - df['Fraction_Remaining']).clip(lower=0.0, upper=1.0) # Ensure y is [0, 1]
    time_scaler = MinMaxScaler(); temp_scaler = MinMaxScaler() # Scale Temp K
    df['Time_scaled'] = time_scaler.fit_transform(df[['Time_days']].values)
    df['Temp_K_scaled'] = temp_scaler.fit_transform(df[['Temperature_K']].values)
    df_ic = df[df['Time_days'] == 0].copy()
    if not df_ic.empty: df_ic['Time_scaled'] = time_scaler.transform(df_ic[['Time_days']].values)
    print(f"Loaded {len(df)} data points for use. Found {len(df_ic)} initial condition points.")
    return df, df_ic, time_scaler, temp_scaler


# =============================================================================
# 2. Neural ODE Model Definition
# =============================================================================
if torch_available:
    class ODEFunc(nn.Module):
        """ Approximates the derivative dy/dt = f(y, T) """
        def __init__(self, n_neurons, n_layers):
            super().__init__()
            # Input: Concatenation of y (1 dim) and T_k_scaled (1 dim) = 2 dims
            layers = [nn.Linear(2, n_neurons), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
            layers.append(nn.Linear(n_neurons, 1)) # Output: dy/dt (1 dim)
            self.net = nn.Sequential(*layers)
            self._temp_k_scaled = None # To store temperature context

        def set_context(self, temp_k_scaled):
            """Set the temperature context for the current ODE solve."""
            # Ensure context is stored with a batch-like dimension [1, 1]
            self._temp_k_scaled = temp_k_scaled.reshape(1, 1)

        def forward(self, t, y):
            """
            Calculate dy/dt.
            t: current time (scalar or tensor, often unused if autonomous)
            y: current state tensor, MUST be shape [batch_size, state_dim] = [batch_size, 1]
            """
            if self._temp_k_scaled is None:
                raise ValueError("Temperature context not set in ODEFunc")
            # Assert input y shape is as expected by torchdiffeq [batch, state_dim]
            if y.ndim != 2 or y.size(1) != 1:
                 raise ValueError(f"ODEFunc expects y shape [batch, 1], got {y.shape}")

            # Expand temperature context to match batch size of y
            temp_expanded = self._temp_k_scaled.expand(y.size(0), -1) # Shape [batch, 1]

            # Concatenate state y [batch, 1] and context T [batch, 1] -> [batch, 2]
            nn_input = torch.cat([y, temp_expanded], dim=1)
            dydt = self.net(nn_input) # Output shape should be [batch, 1]
            return dydt

# =============================================================================
# 3. Training Setup
# =============================================================================
def train_neural_ode():
    """Conceptual Neural ODE training loop with fixes for time monotonicity and shapes."""
    if not torch_available: print("Cannot train: PyTorch/torchdiffeq not available."); return None, None, None
    df_all, df_ic, time_scaler, temp_scaler = load_and_prepare_data(excel_filename, EXCLUDE_TEMPS_C_TRAINING)
    if df_all is None: return None, None, None
    df_train = df_all[df_all['Time_days'] <= max_training_days].copy()
    if df_train.empty: print("ERROR: No data available for training."); return None, None, None

    # --- Prepare grouped data, handling duplicate times ---
    grouped_train = df_train.groupby('Temperature_K')
    train_data_grouped = []
    for temp_k, group in grouped_train:
        group_agg = group.groupby('Time_scaled')['Conversion_y'].mean().reset_index()
        if len(group_agg) < 2: continue
        # Ensure t=0 point exists for y0 (use mean value if multiple t=0 entries)
        if group_agg['Time_scaled'].min() > time_scaler.transform([[0]])[0][0] + 1e-9: # Check if t=0 scaled is present
             print(f"Warning: Temp {temp_k-273.15:.1f}C group missing t=0 data point. Skipping.")
             continue

        t_points = torch.tensor(group_agg['Time_scaled'].values, dtype=torch.float32).to(device)
        y_actual = torch.tensor(group_agg['Conversion_y'].values, dtype=torch.float32).reshape(-1, 1).to(device)
        temp_k_scaled = torch.tensor(group['Temp_K_scaled'].iloc[0], dtype=torch.float32).to(device)
        train_data_grouped.append({'temp_k': temp_k, 'temp_k_scaled': temp_k_scaled, 't': t_points, 'y': y_actual})

    if not train_data_grouped: print("ERROR: No valid temperature groups for training."); return None, None, None

    ode_func = ODEFunc(N_ODE_NEURONS, N_ODE_HIDDEN_LAYERS).to(device)
    optimizer = optim.Adam(ode_func.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting Neural ODE Training ({N_EPOCHS} Epochs) ---"); start_time = time.time(); loss_history = []
    for epoch in range(N_EPOCHS):
        ode_func.train(); total_epoch_loss = 0.0; optimizer.zero_grad()
        for traj_data in train_data_grouped:
            t_eval_points = traj_data['t'] # Unique, sorted time points for this trajectory
            y_target = traj_data['y']      # Target y values corresponding to t_eval_points
            temp_k_scaled_context = traj_data['temp_k_scaled']

            ode_func.set_context(temp_k_scaled_context);
            # --- Ensure y0 has shape [1, 1] ---
            y0 = torch.tensor([[0.0]], dtype=torch.float32).to(device) # Shape [batch=1, state_dim=1]

            # Solve the Neural ODE
            y_pred_solved = odeint(
                ode_func,
                y0,
                t_eval_points, # Pass unique, sorted times
                method=ODE_SOLVER_METHOD,
                rtol=ODE_RTOL,
                atol=ODE_ATOL
            ) # Output shape: [time_points, batch_size=1, state_dim=1]

            # --- Ensure prediction and target shapes match [time, 1] ---
            y_pred = y_pred_solved.squeeze(1) # Remove batch dim -> shape [time, 1]

            if y_pred.shape != y_target.shape:
                 print(f"Warning: Mismatch in prediction ({y_pred.shape}) and target ({y_target.shape}) shapes. Skipping loss.")
                 continue

            loss = torch.mean((y_pred - y_target)**2); total_epoch_loss += loss
        # Average loss over trajectories and backpropagate
        if len(train_data_grouped) > 0:
             avg_loss = total_epoch_loss / len(train_data_grouped)
             # Check for NaN/Inf loss before backward pass
             if not torch.isnan(avg_loss) and not torch.isinf(avg_loss):
                 avg_loss.backward(); optimizer.step(); loss_history.append(avg_loss.item())
             else:
                 print(f"Warning: NaN or Inf loss encountered at epoch {epoch+1}. Skipping backward/step.")
                 loss_history.append(loss_history[-1] if loss_history else np.nan) # Keep last valid loss or NaN

             if (epoch + 1) % 500 == 0: print(f"Epoch [{epoch+1}/{N_EPOCHS}], Avg Loss: {avg_loss.item():.4e}")
        else:
             print("Warning: No trajectories processed in epoch.")


    end_time = time.time(); print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")
    plt.figure(figsize=(8, 5)); plt.plot(loss_history); plt.yscale('log'); plt.title("Neural ODE Training Loss History"); plt.xlabel("Epoch"); plt.ylabel("Average MSE Loss (y)"); plt.grid(True); plt.show()
    return ode_func, time_scaler, temp_scaler

# =============================================================================
# 4. Prediction and Plotting (Conceptual - Neural ODE)
# =============================================================================
# ... (predict_and_plot_neural_ode function needs y0 shape fix) ...
def predict_and_plot_neural_ode(ode_func_trained, time_scaler, temp_scaler, df_actual):
    if not torch_available or ode_func_trained is None: print("Cannot predict..."); return
    if df_actual is None or df_actual.empty: print("Cannot plot..."); return
    ode_func_trained.eval()
    pred_time_days = np.linspace(0, prediction_days, 200); pred_temps_C = sorted(df_actual['Temperature_C'].unique())
    pred_temps_K = [T + 273.15 for T in pred_temps_C]; pred_time_scaled = time_scaler.transform(pred_time_days.reshape(-1, 1))
    t_pred_tensor = torch.tensor(pred_time_scaled, dtype=torch.float32).to(device).flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_temps_C))); temp_color_map = {temp: color for temp, color in zip(pred_temps_C, colors)}
    plt.figure(figsize=(12, 8))
    with torch.no_grad():
        for i, temp_k in enumerate(pred_temps_K):
            temp_c = pred_temps_C[i]; color = temp_color_map.get(temp_c, 'grey')
            purity_t0_series = df_actual[df_actual['Temperature_C'] == temp_c]['Purity_t0']
            if purity_t0_series.empty or pd.isna(purity_t0_series.iloc[0]) or purity_t0_series.iloc[0] <= 0: continue
            purity_t0 = purity_t0_series.iloc[0]
            temp_k_scaled_val = temp_scaler.transform(np.array([[temp_k]]))[0][0]
            temp_k_scaled_context = torch.tensor([temp_k_scaled_val], dtype=torch.float32).to(device)
            ode_func_trained.set_context(temp_k_scaled_context)
            # --- Ensure y0 has shape [1, 1] for prediction ---
            y0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
            y_pred_curve_solved = odeint(ode_func_trained, y0, t_pred_tensor, method=ODE_SOLVER_METHOD, rtol=ODE_RTOL, atol=ODE_ATOL).cpu().numpy()
            # Output shape is [time, batch=1, state=1], need shape [time]
            y_pred_curve = y_pred_curve_solved.squeeze() # Squeeze batch and state dims

            P_pred_curve = (1.0 - y_pred_curve) * purity_t0
            plt.plot(pred_time_days, P_pred_curve, '--', label=f'{temp_c:.0f}°C (NeuralODE Pred)', color=color)
            actual_data_temp = df_actual[df_actual['Temperature_C'] == temp_c]
            if not actual_data_temp.empty: plt.scatter(actual_data_temp['Time_days'], actual_data_temp['Purity'], label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)
    plt.xlabel("Time (days)"); plt.ylabel("Purity (%)"); plt.title("Neural ODE Prediction vs Experimental Data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.7)
    min_purity_exp = df_actual['Purity'].min() if not df_actual.empty and df_actual['Purity'].notna().any() else 0
    max_purity_exp = df_actual['Purity_t0'].max() if not df_actual.empty and 'Purity_t0' in df_actual and df_actual['Purity_t0'].notna().any() else 100
    plt.ylim(bottom=max(0, min_purity_exp - 10), top=max_purity_exp + 2 if pd.notna(max_purity_exp) else 105)
    plt.xlim(left=-prediction_days*0.02, right=prediction_days*1.02); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()


# =============================================================================
# 5. Evaluation Metrics & Saving (Conceptual - Requires adapting previous functions)
# =============================================================================
# Adapt functions from SA-PINN script to use ode_func and odeint for predictions

# =============================================================================
# Main Execution Block (Conceptual)
# =============================================================================
if __name__ == "__main__":
    print("--- Running Neural ODE Conceptual Script (Shape Fix) ---")
    trained_ode_func, t_scaler, T_scaler = train_neural_ode()
    df_orig, _, _, _ = load_and_prepare_data(excel_filename) # Load original data for final plot
    if trained_ode_func and df_orig is not None:
        predict_and_plot_neural_ode(trained_ode_func, t_scaler, T_scaler, df_orig)
        # Add calls to evaluation and saving functions here if implemented
    else:
        print("\nSkipping prediction plotting due to training failure or data loading issues.")
    print("\n--- Conceptual Script Finished ---")

