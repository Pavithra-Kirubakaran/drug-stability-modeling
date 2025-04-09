# -*- coding: utf-8 -*-
"""
Conceptual Implementation of a Self-Adaptive Physics-Informed Neural Network (SA-PINN)
inspired approach for modeling purity degradation using PyTorch.

NOTE: This code requires PyTorch and cannot be run in this environment.

Changes:
- Fixed second SyntaxError in dummy class definitions (DummyTensor methods).
- Kept previous stabilization changes (reduced lambda LR, clipping).
- Kept option for trainable n, m and excluding temps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# --- PyTorch Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    torch_available = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except ImportError:
    print("PyTorch not found. Using dummy classes for conceptual structure.")
    print("Install using: pip install torch pandas openpyxl matplotlib scikit-learn")
    # --- Fixed Dummy Classes ---
    class DummyModule: pass
    class DummyLinear: pass
    class DummyTanh: pass
    class DummySigmoid: pass
    class DummyParam:
        def __init__(self, data): self.data = data; self._grad = None
        def item(self): return self.data.item() if hasattr(self.data, 'item') else self.data
        def detach(self): return self # Simplistic detach
        def backward(self, retain_graph=False): pass # Dummy backward
        def zero_(self): pass # Dummy zero_
        def clamp_(self, min=None, max=None): # Dummy clamp
             if min is not None: self.data = np.maximum(self.data, min)
             if max is not None: self.data = np.minimum(self.data, max)
             return self # Return self for chaining
        @property
        def grad(self): return self._grad
        @grad.setter
        def grad(self, value): self._grad = value

    class DummyNN:
        Module = DummyModule
        Linear = DummyLinear
        Tanh = DummyTanh
        Sigmoid = DummySigmoid
        Parameter = lambda x: DummyParam(x)
        MSELoss = lambda: None

    nn = DummyNN()

    class DummyTensor:
        def __init__(self, data, dtype=None):
            self.data = np.array(data) if not isinstance(data, DummyTensor) else data.data
            self.dtype=dtype
            self._requires_grad = False
        def reshape(self, *args):
            return DummyTensor(self.data.reshape(*args))
        def item(self):
            return self.data.item() if self.data.size == 1 else self.data
        # --- Fixed method definitions ---
        def cpu(self):
            return self # Assume CPU if no torch
        def numpy(self):
            return self.data
        def requires_grad_(self, requires_grad=True):
            self._requires_grad = requires_grad
            return self
        # --- End Fixed method definitions ---
        def detach(self):
            return self # Simplistic detach
        def backward(self, retain_graph=False):
            pass # Dummy backward
        def clamp(self, min=None, max=None): # Dummy clamp (non-inplace)
             new_data = self.data.copy()
             if min is not None: new_data = np.maximum(new_data, min)
             if max is not None: new_data = np.minimum(new_data, max)
             return DummyTensor(new_data)
        def clamp_(self, min=None, max=None): # Dummy clamp (inplace)
             if min is not None: self.data = np.maximum(self.data, min)
             if max is not None: self.data = np.minimum(self.data, max)
             return self
        def __add__(self, other): return DummyTensor(self.data + (other.data if isinstance(other, DummyTensor) else other))
        def __radd__(self, other): return self.__add__(other)
        def __sub__(self, other): return DummyTensor(self.data - (other.data if isinstance(other, DummyTensor) else other))
        def __rsub__(self, other): return DummyTensor((other.data if isinstance(other, DummyTensor) else other) - self.data)
        def __mul__(self, other): return DummyTensor(self.data * (other.data if isinstance(other, DummyTensor) else other))
        def __rmul__(self, other): return self.__mul__(other)
        def __pow__(self, other): return DummyTensor(self.data ** (other.data if isinstance(other, DummyTensor) else other))
        def __neg__(self): return DummyTensor(-self.data)
        def __len__(self): return len(self.data)
        def __getitem__(self, key): return DummyTensor(self.data[key])
        def size(self, dim=None): return self.data.shape if dim is None else self.data.shape[dim]
        def float(self): self.dtype = np.float32; return self # Dummy float conversion
        def fill_(self, value): self.data.fill(value); return self # Dummy fill_


    class DummyOptim:
        def __init__(self, params, lr): self.params = params; self.lr = lr
        def zero_grad(self):
            for p in self.params:
                 if isinstance(p, DummyParam): p.grad = None
        def step(self):
             for p in self.params:
                 if isinstance(p, DummyParam) and p.grad is not None:
                     p.data -= self.lr * p.grad.data # Simplistic SGD step

    class DummyScheduler:
        def __init__(self, optimizer, T_max, eta_min=0): self.optimizer = optimizer; self.base_lr = optimizer.lr
        def step(self): pass # Dummy step
        def get_last_lr(self): return [self.optimizer.lr]

    torch = lambda: None; torch.tensor = DummyTensor; torch.cat = lambda tensors, dim: DummyTensor(np.concatenate([t.data for t in tensors], axis=dim))
    torch.exp = lambda x: DummyTensor(np.exp(x.data)); torch.log = lambda x: DummyTensor(np.log(x.data)); torch.rand = lambda *size, dtype, device: DummyTensor(np.random.rand(*size))
    torch.zeros = lambda *size, dtype, device: DummyTensor(np.zeros(size)); torch.ones_like = lambda x: DummyTensor(np.ones_like(x.data)); torch.full_like = lambda x, val: DummyTensor(np.full_like(x.data, val))
    torch.mean = lambda x: DummyTensor(np.mean(x.data)); torch.clamp = lambda x, min=None, max=None: DummyTensor(np.clip(x.data, a_min=min, a_max=max))
    torch.autograd = lambda: None; torch.autograd.grad = lambda *args, **kwargs: [DummyTensor(np.array([0.0]))]
    torch.no_grad = lambda: type('dummy_no_grad', (), {'__enter__': lambda self: None, '__exit__': lambda *args: None})() # Dummy context manager
    torch.device = lambda x: "cpu"; optim = lambda: None; optim.Adam = DummyOptim; optim.lr_scheduler = lambda: None; optim.lr_scheduler.CosineAnnealingLR = DummyScheduler
    torch_available = False; device = "cpu"
    # --- End Fixed Dummy Classes ---


# =============================================================================
# Configuration & Constants << --- USER TUNING AREA --- >>
# =============================================================================
excel_filename = 'purity_data.xlsx'
predictions_filename = 'purity_predictions_SA_PINN_tuned_3yr.xlsx'

# --- Physics Model Options ---
TRAIN_N_M = False
FIXED_N = 0.5 # << --- UPDATE THIS based on traditional fit results
FIXED_M = 1.0 # << --- UPDATE THIS based on traditional fit results
# Option 2: Make n, m trainable parameters
# TRAIN_N_M = True
INITIAL_N_GUESS = 0.5
INITIAL_M_GUESS = 1.0

# --- Data Options ---
EXCLUDE_TEMPS_C_TRAINING = [5.0] # List of temps (in C) to exclude from training data

# --- PINN Hyperparameters ---
N_HIDDEN_LAYERS = 4       # Try 4-8
N_NEURONS_PER_LAYER = 64  # Try 32, 64, 128
LEARNING_RATE_NET = 1e-4  # Try 1e-3, 5e-4, 1e-4, 5e-5
LEARNING_RATE_LAMBDA = 1e-5 # << --- SIGNIFICANTLY REDUCED (Try 1e-4, 1e-5, 1e-6)
LEARNING_RATE_ORDERS = 1e-4 # LR if training n, m
N_EPOCHS = 50000          # Increase significantly (50k, 100k+)
N_COLLOCATION_POINTS = 5000
INITIAL_LOG_LAMBDA = 0.0  # Initial weight = exp(0) = 1
# --- Lambda Clipping (Log Scale) ---
MIN_LOG_LAMBDA = -6.0 # Corresponds to lambda ~ 0.0025
MAX_LOG_LAMBDA = 9.0  # Corresponds to lambda ~ 8100 (Adjust as needed)

# --- Training/Validation Split ---
max_training_days = 90

# --- Other Constants ---
R_GAS = 8.314
prediction_days = 3 * 365

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================
# ... (load_and_prepare_data function remains the same) ...
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
    df['Conversion_y'] = (1.0 - df['Fraction_Remaining']).clip(lower=0.0, upper=1.0)
    time_scaler = MinMaxScaler(); temp_scaler = MinMaxScaler()
    df['Time_scaled'] = time_scaler.fit_transform(df[['Time_days']].values)
    df['Temp_K_scaled'] = temp_scaler.fit_transform(df[['Temperature_K']].values)
    df_ic = df[df['Time_days'] == 0].copy()
    if not df_ic.empty: df_ic['Time_scaled'] = time_scaler.transform(df_ic[['Time_days']].values)
    print(f"Loaded {len(df)} data points for use. Found {len(df_ic)} initial condition points.")
    return df, df_ic, time_scaler, temp_scaler

# =============================================================================
# 2. PINN Model Definition (Includes optional trainable n, m)
# =============================================================================
# ... (SA_PINN_Autocat class definition remains the same) ...
if torch_available:
    class SA_PINN_Autocat(nn.Module):
        def __init__(self, n_layers, n_neurons, initial_log_lambda=0.0, train_nm=False, n0=0.5, m0=1.0):
            super().__init__()
            self.train_nm = train_nm; layers = [nn.Linear(2, n_neurons), nn.Tanh()]
            for _ in range(n_layers - 1): layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
            layers.append(nn.Linear(n_neurons, 1)); layers.append(nn.Sigmoid()); self.network = nn.Sequential(*layers)
            self.logA = nn.Parameter(torch.tensor([np.log(1e-3)], dtype=torch.float32)); self.logEa_div_R = nn.Parameter(torch.tensor([np.log(80000 / R_GAS)], dtype=torch.float32))
            if self.train_nm: self.n_param = nn.Parameter(torch.tensor([n0], dtype=torch.float32)); self.m_param = nn.Parameter(torch.tensor([m0], dtype=torch.float32))
            else: self.n_param = torch.tensor([n0], dtype=torch.float32); self.m_param = torch.tensor([m0], dtype=torch.float32)
            self.log_lambda_data = nn.Parameter(torch.tensor([initial_log_lambda], dtype=torch.float32)); self.log_lambda_physics = nn.Parameter(torch.tensor([initial_log_lambda], dtype=torch.float32)); self.log_lambda_ic = nn.Parameter(torch.tensor([initial_log_lambda], dtype=torch.float32))
        def forward(self, t_scaled, T_k_scaled): x = torch.cat([t_scaled.reshape(-1, 1), T_k_scaled.reshape(-1, 1)], dim=1); return self.network(x)
        def get_kinetic_params(self): A = torch.exp(self.logA); Ea = torch.exp(self.logEa_div_R) * R_GAS; return A, Ea
        def get_reaction_orders(self):
            if self.train_nm: n = torch.clamp(self.n_param, min=0.0); m = torch.clamp(self.m_param, min=0.0); return n, m
            else: return self.n_param.to(device), self.m_param.to(device)
        def get_adaptive_weights(self): ld = torch.exp(self.log_lambda_data); lp = torch.exp(self.log_lambda_physics); li = torch.exp(self.log_lambda_ic); return ld, lp, li
        def get_network_params(self):
            params = list(self.network.parameters()) + [self.logA, self.logEa_div_R]
            if self.train_nm: params.extend([self.n_param, self.m_param]);
            return params # Return list directly
        def get_lambda_params(self): return [self.log_lambda_data, self.log_lambda_physics, self.log_lambda_ic]

# =============================================================================
# 3. Loss Functions (Autocatalytic Physics)
# =============================================================================
# ... (calculate_autocat_physics_loss function remains the same) ...
if torch_available:
    def calculate_autocat_physics_loss(pinn_model, t_coll_scaled, T_k_scaled_coll, T_k_coll):
        t_coll_scaled.requires_grad_(True); T_k_scaled_coll.requires_grad_(False)
        y_pred = pinn_model(t_coll_scaled, T_k_scaled_coll)
        dy_dt = torch.autograd.grad(y_pred, t_coll_scaled, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        A, Ea = pinn_model.get_kinetic_params(); n, m = pinn_model.get_reaction_orders()
        K = A * torch.exp(-Ea / (R_GAS * T_k_coll.reshape(-1, 1)))
        epsilon = 1e-8; y_safe = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        physics_rate = K * (y_safe**n) * ((1.0 - y_safe)**m)
        residual = dy_dt - physics_rate
        return torch.mean(residual**2)
    mse_loss = nn.MSELoss()

# =============================================================================
# 4. Training Setup (Conceptual PyTorch - SA version with Stabilization)
# =============================================================================
# ... (train_sa_pinn_autocat function remains the same) ...
def train_sa_pinn_autocat():
    if not torch_available: print("Cannot train: PyTorch is not available."); return None, None, None
    df_all, df_ic, time_scaler, temp_scaler = load_and_prepare_data(excel_filename, EXCLUDE_TEMPS_C_TRAINING)
    if df_all is None: return None, None, None
    df_train = df_all[(df_all['Time_days'] > 0) & (df_all['Time_days'] <= max_training_days) & df_all['Conversion_y'].notna()].copy()
    if df_train.empty: print("ERROR: No data available for training."); return None, None, None
    t_train_data = torch.tensor(df_train['Time_scaled'].values, dtype=torch.float32).to(device)
    T_k_scaled_train_data = torch.tensor(df_train['Temp_K_scaled'].values, dtype=torch.float32).to(device)
    y_actual_train_data = torch.tensor(df_train['Conversion_y'].values, dtype=torch.float32).reshape(-1, 1).to(device)
    if df_ic.empty: t_ic, T_k_scaled_ic, y_ic = None, None, None
    else:
        t_ic = torch.tensor(df_ic['Time_scaled'].values, dtype=torch.float32).to(device)
        T_k_scaled_ic = torch.tensor(df_ic['Temp_K_scaled'].values, dtype=torch.float32).to(device)
        y_ic = torch.zeros(len(df_ic), 1, dtype=torch.float32).to(device)
    t_min_scaled = time_scaler.transform([[0]])[0][0]; t_max_scaled = time_scaler.transform([[df_all['Time_days'].max()]])[0][0]
    T_k_min_scaled = temp_scaler.transform([[df_all['Temperature_K'].min()]])[0][0]; T_k_max_scaled = temp_scaler.transform([[df_all['Temperature_K'].max()]])[0][0]
    t_coll_scaled = torch.rand(N_COLLOCATION_POINTS, 1, dtype=torch.float32).to(device) * (t_max_scaled - t_min_scaled) + t_min_scaled
    T_k_scaled_coll_np = np.random.rand(N_COLLOCATION_POINTS, 1) * (T_k_max_scaled - T_k_min_scaled) + T_k_min_scaled
    T_k_scaled_coll = torch.tensor(T_k_scaled_coll_np, dtype=torch.float32).to(device)
    T_k_coll_np = temp_scaler.inverse_transform(T_k_scaled_coll_np); T_k_coll = torch.tensor(T_k_coll_np, dtype=torch.float32).to(device)
    sa_pinn_model = SA_PINN_Autocat(N_HIDDEN_LAYERS, N_NEURONS_PER_LAYER, INITIAL_LOG_LAMBDA, train_nm=TRAIN_N_M, n0=INITIAL_N_GUESS if TRAIN_N_M else FIXED_N, m0=INITIAL_M_GUESS if TRAIN_N_M else FIXED_M).to(device)
    net_params = sa_pinn_model.get_network_params(); lambda_params = sa_pinn_model.get_lambda_params()
    if TRAIN_N_M:
        order_params = [p for p in net_params if p is sa_pinn_model.n_param or p is sa_pinn_model.m_param]
        other_net_params = [p for p in net_params if p is not sa_pinn_model.n_param and p is not sa_pinn_model.m_param]
        optimizer_net = optim.Adam([{'params': other_net_params}, {'params': order_params, 'lr': LEARNING_RATE_ORDERS}], lr=LEARNING_RATE_NET)
    else: optimizer_net = optim.Adam(net_params, lr=LEARNING_RATE_NET)
    optimizer_lambda = optim.Adam(lambda_params, lr=LEARNING_RATE_LAMBDA) # Use reduced LR
    scheduler_net = CosineAnnealingLR(optimizer_net, T_max=N_EPOCHS // 2, eta_min=LEARNING_RATE_NET * 0.01)
    print(f"\n--- Starting SA-PINN Training ({N_EPOCHS} Epochs, Stabilized) ---"); start_time = time.time()
    loss_history = {'total': [], 'data': [], 'physics': [], 'ic': []}; lambda_history = {'data': [], 'physics': [], 'ic': []}; nm_history = {'n': [], 'm': []}
    for epoch in range(N_EPOCHS):
        sa_pinn_model.train()
        y_pred_train_data = sa_pinn_model(t_train_data, T_k_scaled_train_data); loss_d = mse_loss(y_pred_train_data, y_actual_train_data)
        if t_ic is not None: y_pred_ic = sa_pinn_model(t_ic, T_k_scaled_ic); loss_i = mse_loss(y_pred_ic, y_ic)
        else: loss_i = torch.tensor(0.0, device=device)
        loss_p = calculate_autocat_physics_loss(sa_pinn_model, t_coll_scaled, T_k_scaled_coll, T_k_coll)
        lambda_data, lambda_physics, lambda_ic = sa_pinn_model.get_adaptive_weights()
        total_loss_net = (lambda_data.detach() * loss_d + lambda_physics.detach() * loss_p + lambda_ic.detach() * loss_i)
        total_loss_lambda = -(lambda_data * loss_d.detach() + lambda_physics * loss_p.detach() + lambda_ic * loss_i.detach())
        optimizer_net.zero_grad(); total_loss_net.backward(retain_graph=True); optimizer_net.step(); scheduler_net.step()
        optimizer_lambda.zero_grad(); total_loss_lambda.backward(); optimizer_lambda.step()
        with torch.no_grad():
            for log_lambda in sa_pinn_model.get_lambda_params(): log_lambda.clamp_(min=MIN_LOG_LAMBDA, max=MAX_LOG_LAMBDA)
        loss_history['total'].append(total_loss_net.item()); loss_history['data'].append(loss_d.item()); loss_history['physics'].append(loss_p.item()); loss_history['ic'].append(loss_i.item())
        lambda_history['data'].append(lambda_data.item()); lambda_history['physics'].append(lambda_physics.item()); lambda_history['ic'].append(lambda_ic.item())
        n_curr, m_curr = sa_pinn_model.get_reaction_orders(); nm_history['n'].append(n_curr.item()); nm_history['m'].append(m_curr.item())
        if (epoch + 1) % 1000 == 0:
            A_curr, Ea_curr = sa_pinn_model.get_kinetic_params(); ld, lp, li = sa_pinn_model.get_adaptive_weights(); n_disp, m_disp = sa_pinn_model.get_reaction_orders()
            print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss(Net): {total_loss_net.item():.4e} (Ld={loss_d.item():.3e}, Lp={loss_p.item():.3e}, Li={loss_i.item():.3e}) | Lambdas(d,p,i): {ld.item():.2f}, {lp.item():.2f}, {li.item():.2f} | A: {A_curr.item():.2e}, Ea: {Ea_curr.item()/1000:.1f} kJ/mol, n: {n_disp.item():.2f}, m: {m_disp.item():.2f} | LR(Net): {scheduler_net.get_last_lr()[0]:.1e}")
    end_time = time.time(); print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")
    fig, ax1 = plt.subplots(figsize=(10, 6)); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (Log Scale)', color='tab:red')
    ax1.plot(loss_history['total'], color='tab:red', label='Total Loss (for Net)'); ax1.plot(loss_history['data'], color='tab:blue', linestyle=':', alpha=0.7, label='Data Loss (Unweighted)')
    ax1.plot(loss_history['physics'], color='tab:green', linestyle=':', alpha=0.7, label='Physics Loss (Unweighted)'); ax1.plot(loss_history['ic'], color='tab:orange', linestyle=':', alpha=0.7, label='IC Loss (Unweighted)')
    ax1.set_yscale('log'); ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.legend(loc='upper left'); ax1.grid(True)
    ax2 = ax1.twinx(); ax2.set_ylabel('Lambda Weights', color='tab:purple'); ax2.plot(lambda_history['data'], color='blue', linestyle='--', label='Lambda Data')
    ax2.plot(lambda_history['physics'], color='green', linestyle='--', label='Lambda Physics'); ax2.plot(lambda_history['ic'], color='orange', linestyle='--', label='Lambda IC')
    ax2.tick_params(axis='y', labelcolor='tab:purple'); ax2.legend(loc='upper right'); fig.tight_layout(); plt.title("SA-PINN Training History - Loss & Lambdas"); plt.show()
    if TRAIN_N_M: plt.figure(figsize=(10, 4)); plt.plot(nm_history['n'], label='n (Trained)'); plt.plot(nm_history['m'], label='m (Trained)'); plt.xlabel("Epoch"); plt.ylabel("Reaction Order"); plt.title("Trained Reaction Orders History"); plt.legend(); plt.grid(True); plt.show()
    return sa_pinn_model, time_scaler, temp_scaler

# =============================================================================
# 5. Generate Predictions for Evaluation
# =============================================================================
# ... (generate_predictions_for_eval function remains the same) ...
def generate_predictions_for_eval(pinn_model, time_scaler, temp_scaler, df_eval_input):
    if not torch_available or pinn_model is None: return None
    if df_eval_input is None or df_eval_input.empty: return None
    pinn_model.eval(); df_eval = df_eval_input.copy(); df_eval['Predicted_Purity'] = np.nan; df_eval['Predicted_Conversion_y'] = np.nan
    A_final, Ea_final = pinn_model.get_kinetic_params(); n_final, m_final = pinn_model.get_reaction_orders()
    A_final, Ea_final = A_final.item(), Ea_final.item(); n_final, m_final = n_final.item(), m_final.item()
    print(f"Using final parameters for evaluation: A={A_final:.2e}, Ea={Ea_final/1000:.1f}, n={n_final:.3f}, m={m_final:.3f}")
    t_eval_scaled = torch.tensor(df_eval['Time_scaled'].values, dtype=torch.float32).reshape(-1,1).to(device)
    T_k_scaled_eval = torch.tensor(df_eval['Temp_K_scaled'].values, dtype=torch.float32).reshape(-1,1).to(device)
    with torch.no_grad(): y_pred_eval = pinn_model(t_eval_scaled, T_k_scaled_eval).cpu().numpy().flatten()
    df_eval['Predicted_Conversion_y'] = y_pred_eval
    valid_p0_idx_eval = df_eval['Purity_t0'].notna() & (df_eval['Purity_t0'] > 0)
    df_eval.loc[valid_p0_idx_eval, 'Predicted_Purity'] = (1.0 - df_eval.loc[valid_p0_idx_eval, 'Predicted_Conversion_y']) * df_eval.loc[valid_p0_idx_eval, 'Purity_t0']
    df_eval_out = df_eval.dropna(subset=['Predicted_Purity', 'Purity']).copy()
    print(f"Generated predictions for {len(df_eval_out)} evaluation points.")
    return df_eval_out

# =============================================================================
# 6. Calculate Evaluation Metrics (Train vs Validation)
# =============================================================================
# ... (calculate_and_print_metrics function remains the same) ...
def calculate_and_print_metrics(df_eval_predicted):
    if df_eval_predicted is None or df_eval_predicted.empty: print("No predictions available for metric calculation."); return
    print("\n--- Model Evaluation Metrics (Based on Purity Predictions) ---")
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred); non_zero_idx = y_true != 0
        if np.sum(non_zero_idx) == 0: return np.nan
        return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
    eval_metrics = df_eval_predicted[df_eval_predicted['Time_days'] > 0]
    eval_train = eval_metrics[eval_metrics['Time_days'] <= max_training_days]
    eval_valid = eval_metrics[eval_metrics['Time_days'] > max_training_days]
    metrics_results = {}
    for label, df_subset in [('Training (0 < t <= 90 days)', eval_train), ('Validation (t > 90 days)', eval_valid), ('Overall (t > 0)', eval_metrics)]:
        if len(df_subset) > 0:
            y_true = df_subset['Purity']; y_pred = df_subset['Predicted_Purity']
            mae = mean_absolute_error(y_true, y_pred); rmse = np.sqrt(mean_squared_error(y_true, y_pred)); r2 = r2_score(y_true, y_pred); mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics_results[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_subset)}
            print(f"\nMetrics for {label} Data (N={len(df_subset)}):"); print(f"  MAE:   {mae:.3f} % Purity"); print(f"  RMSE:{rmse:.3f} % Purity"); print(f"  R2:              {r2:.4f}"); print(f"  MAPE:{mape:.2f} %")
        else: print(f"\nNo data available for {label} set evaluation.")
    print("-" * 40)

# =============================================================================
# 7. Residual Plot
# =============================================================================
# ... (plot_residuals function remains the same) ...
def plot_residuals(df_eval_predicted):
    if df_eval_predicted is None or df_eval_predicted.empty: return
    print("--- Generating Residual Plot (Based on Purity) ---")
    df_eval_predicted['Residual'] = df_eval_predicted['Purity'] - df_eval_predicted['Predicted_Purity']
    plt.figure(figsize=(10, 6)); plt.scatter(df_eval_predicted['Predicted_Purity'], df_eval_predicted['Residual'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--'); plt.xlabel("Predicted Purity (%)"); plt.ylabel("Residual (Actual - Predicted Purity %)")
    plt.title("Residual Plot (SA-PINN Autocatalytic Model)"); plt.grid(True, linestyle=':', alpha=0.7); plt.show(); print("-" * 40)

# =============================================================================
# 8. Extrapolation and Prediction Plot (up to 3 years)
# =============================================================================
# ... (predict_and_plot_autocat function remains the same) ...
def predict_and_plot_autocat(pinn_model, time_scaler, temp_scaler, df_actual):
    if not torch_available or pinn_model is None: print("Cannot predict..."); return
    if df_actual is None or df_actual.empty: print("Cannot plot..."); return
    pinn_model.eval()
    pred_time_days = np.linspace(0, prediction_days, 200); pred_temps_C = sorted(df_actual['Temperature_C'].unique())
    pred_temps_K = [T + 273.15 for T in pred_temps_C]; pred_time_scaled = time_scaler.transform(pred_time_days.reshape(-1, 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_temps_C))); temp_color_map = {temp: color for temp, color in zip(pred_temps_C, colors)}
    plt.figure(figsize=(12, 8))
    with torch.no_grad():
        for i, temp_k in enumerate(pred_temps_K):
            temp_c = pred_temps_C[i]; color = temp_color_map.get(temp_c, 'grey')
            purity_t0_series = df_actual[df_actual['Temperature_C'] == temp_c]['Purity_t0']
            if purity_t0_series.empty or pd.isna(purity_t0_series.iloc[0]) or purity_t0_series.iloc[0] <= 0: print(f"Warning: Skipping prediction for Temp {temp_c}°C due to missing/invalid Purity_t0."); continue
            purity_t0 = purity_t0_series.iloc[0]; t_input = torch.tensor(time_export_scaled, dtype=torch.float32).to(device); temp_k_scaled_val = temp_scaler.transform(np.array([[temp_k]]))[0][0]
            T_k_scaled_input = torch.full_like(t_input, temp_k_scaled_val).to(device); y_pred_curve = pinn_model(t_input, T_k_scaled_input).cpu().numpy().flatten()
            purity_pred_curve = (1.0 - y_pred_curve) * purity_t0
            plt.plot(pred_time_days, P_pred_curve, '--', label=f'{temp_c:.0f}°C (SA-PINN Pred)', color=color)
            actual_data_temp = df_actual[df_actual['Temperature_C'] == temp_c]
            if not actual_data_temp.empty: plt.scatter(actual_data_temp['Time_days'], actual_data_temp['Purity'], label=f'{temp_c:.0f}°C (Exp)', s=30, alpha=0.8, color=color)
    plt.xlabel("Time (days)"); plt.ylabel("Purity (%)"); plt.title("SA-PINN Prediction (Autocatalytic Physics) vs Experimental Data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.7)
    min_purity_exp = df_actual['Purity'].min() if not df_actual.empty and df_actual['Purity'].notna().any() else 0
    max_purity_exp = df_actual['Purity_t0'].max() if not df_actual.empty and 'Purity_t0' in df_actual and df_actual['Purity_t0'].notna().any() else 100
    plt.ylim(bottom=max(0, min_purity_exp - 10), top=max_purity_exp + 2 if pd.notna(max_purity_exp) else 105)
    plt.xlim(left=-prediction_days*0.02, right=prediction_days*1.02); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()


# =============================================================================
# 9. Save Predictions to Excel
# =============================================================================
# ... (save_predictions function remains the same) ...
def save_predictions(pinn_model, time_scaler, temp_scaler, df_actual):
    if not torch_available or pinn_model is None: print("Cannot save predictions..."); return
    if df_actual is None or df_actual.empty: print("Cannot save predictions..."); return
    print(f"\n--- Generating and Saving Predictions up to {prediction_days} days ---")
    pinn_model.eval(); prediction_results_export = []
    time_export = np.linspace(0, prediction_days, int(prediction_days / 5) + 1); time_export_scaled = time_scaler.transform(time_export.reshape(-1, 1))
    pred_temps_C = sorted(df_actual['Temperature_C'].unique()); pred_temps_K = [T + 273.15 for T in pred_temps_C]
    A_final, Ea_final = pinn_model.get_kinetic_params(); n_final, m_final = pinn_model.get_reaction_orders()
    A_final, Ea_final = A_final.item(), Ea_final.item(); n_final, m_final = n_final.item(), m_final.item()
    with torch.no_grad():
        for i, temp_k in enumerate(pred_temps_K):
            temp_c = pred_temps_C[i]; purity_t0_series = df_actual[df_actual['Temperature_C'] == temp_c]['Purity_t0']
            if purity_t0_series.empty or pd.isna(purity_t0_series.iloc[0]) or purity_t0_series.iloc[0] <= 0: continue
            purity_t0 = purity_t0_series.iloc[0]; K_pred = A_final * np.exp(-Ea_final / (R_GAS * temp_k))
            t_input = torch.tensor(time_export_scaled, dtype=torch.float32).to(device); temp_k_scaled_val = temp_scaler.transform(np.array([[temp_k]]))[0][0]
            T_k_scaled_input = torch.full_like(t_input, temp_k_scaled_val).to(device); y_pred_curve = pinn_model(t_input, T_k_scaled_input).cpu().numpy().flatten()
            purity_pred_curve = (1.0 - y_pred_curve) * purity_t0
            for t, y_p, p_p in zip(time_export, y_pred_curve, purity_pred_curve):
                prediction_results_export.append({'Experiment': df_actual['Experiment'].iloc[0], 'Temperature_C': temp_c, 'Time_days': t, 'Predicted_Conversion_y': y_p, 'Predicted_Purity': p_p, 'k_pred_day^-1': K_pred, 'n_used': n_final, 'm_used': m_final})
    if prediction_results_export:
        df_predictions = pd.DataFrame(prediction_results_export); df_predictions = df_predictions.sort_values(by=['Experiment', 'Temperature_C', 'Time_days'])
        try: df_predictions.to_excel(predictions_filename, index=False, engine='openpyxl'); print(f"\n--- Predictions saved to '{predictions_filename}' ---")
        except Exception as e: print(f"\nERROR: Could not save predictions: {e}")
    else: print("\nNo predictions were generated for export.")
    print("-" * 40)

# =============================================================================
# Main Execution Block (Conceptual)
# =============================================================================
if __name__ == "__main__":
    print("--- Running SA-PINN Conceptual Script (Autocatalytic - Stabilized) ---")
    trained_model, t_scaler, T_scaler = train_sa_pinn_autocat()
    df_orig, _, _, _ = load_and_prepare_data(excel_filename) # Load original data *without* exclusions for final eval/plot
    if trained_model and df_orig is not None:
        df_eval_results = generate_predictions_for_eval(trained_model, t_scaler, T_scaler, df_orig)
        if df_eval_results is not None and not df_eval_results.empty:
            calculate_and_print_metrics(df_eval_results)
            plot_residuals(df_eval_results)
        else: print("Skipping metrics and residual plot due to prediction failure.")
        predict_and_plot_autocat(trained_model, t_scaler, T_scaler, df_orig)
        if df_eval_results is not None and not df_eval_results.empty:
             save_predictions(trained_model, t_scaler, T_scaler, df_orig)
    else: print("\nSkipping prediction, evaluation, and plotting.")
    print("\n--- Conceptual Script Finished ---")
