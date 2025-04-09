# -*- coding: utf-8 -*-
"""
Conceptual Implementation of a Physics-Informed Neural Network (PINN)
for modeling purity degradation using PyTorch.

NOTE: This code requires PyTorch and cannot be run in this environment.
      It serves as a structural example of the PINN approach described
      in the 'sciml_explanation' artifact.

Problem: Model Purity(t, T) based on experimental data, incorporating
         the physics assumption dy/dt = K(T) * y^n * (1-y)^m,
         where K(T) = A*exp(-Ea/RT) and y = 1 - Purity/Purity_t0.

Inputs to NN: Time (t), Temperature (T in Kelvin) - Scaled
Output of NN: Conversion (y) - Scaled between 0 and 1
Trainable Parameters: NN weights/biases, logA, logEa_div_R
Fixed Parameters: n, m (reaction orders, assumed known or averaged from prior analysis)

Changes:
- Improved robustness in scaler fitting/transforming.
- Added checks for data validity within prediction/plotting loop.
- Safer calculation of plot limits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import os

# --- PyTorch Imports (Required if running locally) ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # from torch.utils.data import Dataset, DataLoader, TensorDataset # Not used in this simplified structure
except ImportError:
    print("PyTorch not found. This script requires PyTorch to be installed.")
    print("Install using: pip install torch pandas openpyxl matplotlib scikit-learn")
    class nn: # Dummy classes if torch is missing
        class Module: pass
        class Linear: pass
        class Tanh: pass
        class Sigmoid: pass
        Parameter = lambda x: x
    torch_available = False
else:
    torch_available = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# =============================================================================
# Configuration & Constants
# =============================================================================
excel_filename = 'purity_data.xlsx'
predictions_filename = 'purity_predictions_autocat_3yr.xlsx'
# --- Assumed Fixed Reaction Orders (Update if different values were found) ---
FIXED_N = 0.5 # Example value, replace with avg n found previously
FIXED_M = 1.0 # Example value, replace with avg m found previously
# --- PINN Hyperparameters ---
N_HIDDEN_LAYERS = 4
N_NEURONS_PER_LAYER = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 20000 # Increased epochs might be needed
N_COLLOCATION_POINTS = 5000
# --- Loss weights (CRUCIAL - Require tuning!) ---
W_DATA = 1.0     # Weight for matching conversion y data
W_PHYSICS = 0.1  # Weight for the ODE residual (needs tuning)
W_IC = 1.0     # Weight for matching initial conversion y=0

R_GAS = 8.314 # J/mol/K
prediction_days = 3 * 365 # 3 years

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================
def load_and_prepare_data(filename):
    """Loads data, calculates conversion y, and scales inputs."""
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        print(f"ERROR: Excel file '{filename}' not found.")
        return None, None, None, None

    df = df.dropna(subset=['Experiment', 'Temperature_C', 'Time_days', 'Purity'])
    df = df.astype({'Temperature_C': float, 'Time_days': float, 'Purity': float})
    df = df[df['Purity'] >= 0]
    if df.empty:
        print("ERROR: No valid data found after loading and initial cleaning.")
        return None, None, None, None

    df['Temperature_K'] = df['Temperature_C'] + 273.15

    # Calculate Purity_t0 and Fraction_Remaining
    df['Time_rank'] = df.groupby(['Experiment', 'Temperature_C'])['Time_days'].rank(method='first')
    purity_t0_map = df.loc[df['Time_rank'] == 1].set_index(['Experiment', 'Temperature_C'])['Purity']
    df['Purity_t0'] = df.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map)
    df['Purity_t0'] = df['Purity_t0'].fillna(method='ffill').fillna(method='bfill')
    df['Purity_t0'] = df['Purity_t0'].replace(0, np.nan)

    df['Fraction_Remaining'] = np.nan
    valid_p0_idx = df['Purity_t0'].notna() & (df['Purity_t0'] > 0)
    df.loc[valid_p0_idx, 'Fraction_Remaining'] = (df.loc[valid_p0_idx, 'Purity'] / df.loc[valid_p0_idx, 'Purity_t0'])
    df['Fraction_Remaining'] = df['Fraction_Remaining'].fillna(1.0)
    df['Fraction_Remaining'] = df['Fraction_Remaining'].clip(upper=1.0)

    # Calculate Conversion y (alpha) = 1 - Fraction_Remaining
    df['Conversion_y'] = 1.0 - df['Fraction_Remaining']
    df['Conversion_y'] = df['Conversion_y'].clip(lower=0.0, upper=1.0) # Ensure y is [0, 1]

    # Scale Time and Temperature inputs using NumPy arrays
    time_scaler = MinMaxScaler()
    temp_scaler = MinMaxScaler()
    # Fit scalers on the .values to avoid feature name issues
    df['Time_scaled'] = time_scaler.fit_transform(df[['Time_days']].values)
    df['Temp_K_scaled'] = temp_scaler.fit_transform(df[['Temperature_K']].values)

    # Get initial conditions data (t=0, should have y=0)
    df_ic = df[df['Time_days'] == 0].copy()
    if not df_ic.empty:
      # Ensure IC has scaled time corresponding to 0
      df_ic['Time_scaled'] = time_scaler.transform(df_ic[['Time_days']].values)


    print(f"Loaded {len(df)} data points.")
    print(f"Found {len(df_ic)} initial condition points (expect y=0).")
    return df, df_ic, time_scaler, temp_scaler

# =============================================================================
# 2. PINN Model Definition (Outputting Conversion y)
# =============================================================================
if torch_available:
    class PINN_Autocat(nn.Module):
        def __init__(self, n_layers, n_neurons):
            super().__init__()
            # ... (rest of class definition is unchanged) ...
            layers = [nn.Linear(2, n_neurons), nn.Tanh()] # Input: [t_scaled, T_k_scaled]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
            layers.append(nn.Linear(n_neurons, 1))
            layers.append(nn.Sigmoid()) # Output y is [0, 1]
            self.network = nn.Sequential(*layers)
            self.logA = nn.Parameter(torch.tensor([np.log(1e-3)], dtype=torch.float32))
            self.logEa_div_R = nn.Parameter(torch.tensor([np.log(80000 / R_GAS)], dtype=torch.float32))

        def forward(self, t_scaled, T_k_scaled):
            x = torch.cat([t_scaled.reshape(-1, 1), T_k_scaled.reshape(-1, 1)], dim=1)
            conversion_y_pred = self.network(x)
            return conversion_y_pred

        def get_kinetic_params(self):
            A = torch.exp(self.logA)
            Ea = torch.exp(self.logEa_div_R) * R_GAS
            return A, Ea

# =============================================================================
# 3. Loss Functions (Autocatalytic Physics)
# =============================================================================
if torch_available:
    def calculate_autocat_physics_loss(pinn_model, t_coll_scaled, T_k_scaled_coll, T_k_coll, n, m):
        """Calculates the physics residual loss: dy/dt - K(T)*y^n*(1-y)^m = 0"""
        # ... (function definition is unchanged) ...
        t_coll_scaled.requires_grad_(True)
        T_k_scaled_coll.requires_grad_(False)
        y_pred = pinn_model(t_coll_scaled, T_k_scaled_coll)
        dy_dt = torch.autograd.grad(y_pred, t_coll_scaled, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        A, Ea = pinn_model.get_kinetic_params()
        K = A * torch.exp(-Ea / (R_GAS * T_k_coll.reshape(-1, 1)))
        epsilon = 1e-8
        y_safe = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        physics_rate = K * (y_safe**n) * ((1.0 - y_safe)**m)
        residual = dy_dt - physics_rate
        loss_p = torch.mean(residual**2)
        return loss_p

    mse_loss = nn.MSELoss()

# =============================================================================
# 4. Training Setup (Conceptual PyTorch)
# =============================================================================
def train_pinn_autocat():
    """Conceptual PINN training loop for the autocatalytic model."""
    if not torch_available:
        print("Cannot train: PyTorch is not available.")
        return None, None, None

    # --- Load and Prepare Data ---
    df_all, df_ic, time_scaler, temp_scaler = load_and_prepare_data(excel_filename)
    if df_all is None: return None, None, None

    # Convert data to PyTorch tensors
    t_data = torch.tensor(df_all['Time_scaled'].values, dtype=torch.float32).to(device)
    T_k_scaled_data = torch.tensor(df_all['Temp_K_scaled'].values, dtype=torch.float32).to(device)
    y_actual_data = torch.tensor(df_all['Conversion_y'].values, dtype=torch.float32).reshape(-1, 1).to(device)

    # Ensure IC data is valid before creating tensors
    if df_ic.empty:
        print("Warning: No initial condition data points found (Time_days == 0). IC loss cannot be calculated.")
        t_ic, T_k_scaled_ic, y_ic = None, None, None
    else:
        t_ic = torch.tensor(df_ic['Time_scaled'].values, dtype=torch.float32).to(device)
        T_k_scaled_ic = torch.tensor(df_ic['Temp_K_scaled'].values, dtype=torch.float32).to(device)
        y_ic = torch.zeros(len(df_ic), 1, dtype=torch.float32).to(device) # Target y=0 at t=0

    # Collocation points generation (using .values for consistency)
    t_min_scaled = time_scaler.transform([[0]])[0][0]
    t_max_scaled = time_scaler.transform([[df_all['Time_days'].max()]])[0][0]
    T_k_min_scaled = temp_scaler.transform([[df_all['Temperature_K'].min()]])[0][0]
    T_k_max_scaled = temp_scaler.transform([[df_all['Temperature_K'].max()]])[0][0]

    t_coll_scaled = torch.rand(N_COLLOCATION_POINTS, 1, dtype=torch.float32).to(device) * (t_max_scaled - t_min_scaled) + t_min_scaled
    T_k_scaled_coll_np = np.random.rand(N_COLLOCATION_POINTS, 1) * (T_k_max_scaled - T_k_min_scaled) + T_k_min_scaled
    T_k_scaled_coll = torch.tensor(T_k_scaled_coll_np, dtype=torch.float32).to(device)
    # Inverse transform requires NumPy array with correct shape
    T_k_coll_np = temp_scaler.inverse_transform(T_k_scaled_coll_np)
    T_k_coll = torch.tensor(T_k_coll_np, dtype=torch.float32).to(device)


    # --- Initialize Model and Optimizer ---
    pinn_model = PINN_Autocat(N_HIDDEN_LAYERS, N_NEURONS_PER_LAYER).to(device)
    optimizer = optim.Adam(pinn_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    # --- Training Loop ---
    print("\n--- Starting PINN Training (Autocatalytic Physics) ---")
    start_time = time.time()
    loss_history = []

    for epoch in range(N_EPOCHS):
        pinn_model.train()

        # Forward pass for data points
        y_pred_data = pinn_model(t_data, T_k_scaled_data)
        loss_d = mse_loss(y_pred_data, y_actual_data) # Match conversion y

        # Forward pass and loss for IC points (if available)
        if t_ic is not None:
            y_pred_ic = pinn_model(t_ic, T_k_scaled_ic)
            loss_i = mse_loss(y_pred_ic, y_ic) # Match y=0 at t=0
        else:
            loss_i = torch.tensor(0.0).to(device) # No IC loss if no t=0 data

        # Physics loss at collocation points
        loss_p = calculate_autocat_physics_loss(
            pinn_model, t_coll_scaled, T_k_scaled_coll, T_k_coll, FIXED_N, FIXED_M
        )

        # Total weighted loss
        total_loss = W_DATA * loss_d + W_PHYSICS * loss_p + W_IC * loss_i
        loss_history.append(total_loss.item())

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Print progress
        if (epoch + 1) % 1000 == 0:
            A_curr, Ea_curr = pinn_model.get_kinetic_params()
            print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {total_loss.item():.4e} "
                  f"(Data(y): {loss_d.item():.3e}, Physics: {loss_p.item():.3e}, IC(y): {loss_i.item():.3e}) | "
                  f"A: {A_curr.item():.2e}, Ea: {Ea_curr.item()/1000:.1f} kJ/mol | "
                  f"LR: {scheduler.get_last_lr()[0]:.1e}")

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")

    # Plot loss history
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("PINN Training Loss History (Autocatalytic)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.show()

    return pinn_model, time_scaler, temp_scaler

# =============================================================================
# 5. Prediction and Plotting (Conceptual PyTorch - Autocatalytic)
# =============================================================================
def predict_and_plot_autocat(pinn_model, time_scaler, temp_scaler, df_actual):
    """Uses trained PINN (autocat) to predict and plots results."""
    if not torch_available or pinn_model is None:
        print("Cannot predict: PyTorch not available or model not trained.")
        return
    if df_actual is None or df_actual.empty:
        print("Cannot plot: No actual data provided.")
        return

    pinn_model.eval() # Set model to evaluation mode

    # Time and Temp ranges for prediction plots
    pred_time_days = np.linspace(0, prediction_days, 200) # Predict up to 3 years
    pred_temps_C = sorted(df_actual['Temperature_C'].unique())
    pred_temps_K = [T + 273.15 for T in pred_temps_C]

    # Scale prediction inputs using .values
    pred_time_scaled = time_scaler.transform(pred_time_days.reshape(-1, 1))

    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_temps_C)))
    temp_color_map = {temp: color for temp, color in zip(pred_temps_C, colors)}

    plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for i, temp_k in enumerate(pred_temps_K):
            temp_c = pred_temps_C[i]
            color = temp_color_map.get(temp_c, 'grey')

            # Find corresponding Purity_t0 for this condition robustly
            purity_t0_series = df_actual[df_actual['Temperature_C'] == temp_c]['Purity_t0']
            if purity_t0_series.empty or pd.isna(purity_t0_series.iloc[0]) or purity_t0_series.iloc[0] <= 0:
                print(f"Warning: Skipping prediction for Temp {temp_c}°C due to missing/invalid Purity_t0.")
                continue
            purity_t0 = purity_t0_series.iloc[0]

            # Prepare input tensors
            t_input = torch.tensor(pred_time_scaled, dtype=torch.float32).to(device)
            # Scale temperature using .values
            temp_k_scaled_val = temp_scaler.transform(np.array([[temp_k]]))[0][0]
            T_k_scaled_input = torch.full_like(t_input, temp_k_scaled_val).to(device)

            # Predict conversion y
            y_pred_curve = pinn_model(t_input, T_k_scaled_input).cpu().numpy().flatten()
            # Convert y back to Purity
            P_pred_curve = (1.0 - y_pred_curve) * purity_t0

            # Plot prediction curve
            plt.plot(pred_time_days, P_pred_curve, '--', label=f'{temp_c:.0f}°C (PINN Pred)', color=color)

            # Plot actual data for this temperature
            actual_data_temp = df_actual[df_actual['Temperature_C'] == temp_c]
            if not actual_data_temp.empty:
                plt.scatter(actual_data_temp['Time_days'], actual_data_temp['Purity'], label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)

    # Finalize plot
    plt.xlabel("Time (days)")
    plt.ylabel("Purity (%)")
    plt.title("PINN Prediction (Autocatalytic Physics) vs Experimental Data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.7)

    # Safer calculation of plot limits
    min_purity_exp = df_actual['Purity'].min() if not df_actual.empty and df_actual['Purity'].notna().any() else 0
    max_purity_exp = df_actual['Purity_t0'].max() if not df_actual.empty and 'Purity_t0' in df_actual and df_actual['Purity_t0'].notna().any() else 100
    plt.ylim(bottom=max(0, min_purity_exp - 10), top=max_purity_exp + 2 if pd.notna(max_purity_exp) else 105)
    plt.xlim(left=-prediction_days*0.02, right=prediction_days*1.02) # Show full 3 years
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# =============================================================================
# Main Execution Block (Conceptual)
# =============================================================================
if __name__ == "__main__":
    # This block would only run if executed as a script locally with PyTorch
    print("--- Running PINN Conceptual Script (Autocatalytic - Robustness Fixes) ---")

    # Train the model (conceptual)
    trained_model, t_scaler, T_scaler = train_pinn_autocat()

    # Load original data again for plotting comparison
    df_orig, _, _, _ = load_and_prepare_data(excel_filename)

    # Predict and plot if training was successful (conceptual)
    if trained_model and df_orig is not None:
         predict_and_plot_autocat(trained_model, t_scaler, T_scaler, df_orig)
    else:
         print("\nSkipping prediction plotting.")

    print("\n--- Conceptual Script Finished ---")

