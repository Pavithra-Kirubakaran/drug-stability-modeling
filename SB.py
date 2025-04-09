# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:01:56 2025

@author: pavit
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Using numpy for data handling before converting to jax arrays
from typing import Tuple, Dict
import os # Import os to handle file paths

# --- Configuration ---
script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
data_dir = os.path.join(script_dir, 'Data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir) # Create 'Data' directory if it doesn't exist
EXCEL_FILE_PATH = os.path.join(data_dir, 'stability_data.xlsx')

ACCELERATED_DATA_MAX_DAYS = 97
PREDICTION_MAX_DAYS = 3 * 365
SEED = 123
# --- Hyperparameters to Tune ---
INITIAL_LEARNING_RATE = 1e-3
EPOCHS = 50000 # Increase epochs further for more complex model
PHYSICS_LOSS_WEIGHT = 1e-4 # Start lower, SB residual scale might differ
WIDTH_SIZE = 64
DEPTH = 4
LR_DECAY_RATE = 0.95 # Slower decay might be needed
LR_DECAY_STEPS = 2000
GRAD_CLIP_NORM = 1.0
# --- End Hyperparameters ---
R = 8.314e-3 # Gas constant in kJ/(mol*K)

# --- Data Loading and Preparation ---
# (Keep the load_and_prepare_data function exactly the same as the previous version)
def load_and_prepare_data(file_path: str, max_train_days: int) -> Tuple:
    """Loads data from Excel, splits, normalizes inputs, and converts to JAX arrays."""
    if not os.path.exists(file_path):
         print(f"Error: Excel file not found at the specified path: {os.path.abspath(file_path)}")
         print(f"Please place 'stability_data.xlsx' in the '{os.path.basename(data_dir)}' subfolder.")
         raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from {file_path}")
    except Exception as e:
        print(f"Error loading or reading Excel file {file_path}: {e}")
        raise

    required_cols = ['Temperature', 'Time', 'HMWP']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    df.dropna(subset=required_cols, inplace=True)
    if df.empty:
        raise ValueError("Dataframe is empty after dropping NaN values. Check your Excel file.")

    df['Temperature_K'] = df['Temperature'] + 273.15
    df_train = df[df['Time'] <= max_train_days].copy()
    df_val = df[df['Time'] > max_train_days].copy()
    original_data = {'train': df_train.copy(), 'val': df_val.copy()}

    if df_train.empty:
        raise ValueError(f"No training data found with Time <= {max_train_days} days.")

    # --- Normalization ---
    scalers = {
        'Time': {'min': df_train['Time'].min(), 'max': df_train['Time'].max()},
        'Temperature_K': {'min': df_train['Temperature_K'].min(), 'max': df_train['Temperature_K'].max()},
        'HMWP_observed': {'min': df['HMWP'].min(), 'max': df['HMWP'].max()}
    }
    print("Scalers calculated based on training data (Inputs) & all data (HMWP range):", scalers)

    def normalize_inputs(df, scalers):
        df_norm = df.copy()
        for col in ['Time', 'Temperature_K']:
            scale = scalers[col]
            min_val, max_val = scale['min'], scale['max']
            if max_val - min_val == 0:
                 df_norm[col] = 0.0
            else:
                 df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        return df_norm

    df_train_norm_inputs = normalize_inputs(df_train, scalers)
    t_train_norm = jnp.array(df_train_norm_inputs['Time'].values[:, None])
    T_train_norm = jnp.array(df_train_norm_inputs['Temperature_K'].values[:, None])
    hmwp_train_unnorm = jnp.array(df_train['HMWP'].values[:, None])
    X_train_norm = jnp.hstack([t_train_norm, T_train_norm])
    T_train_K_unnorm = jnp.array(df_train['Temperature_K'].values[:, None])

    init_indices = np.where(df_train['Time'].values == 0.0)[0]
    if len(init_indices) == 0:
         print("Warning: No data points found at Time = 0 for initial condition loss.")
         X_init_norm = jnp.empty((0, X_train_norm.shape[1]))
         hmwp_init_unnorm = jnp.empty((0, hmwp_train_unnorm.shape[1]))
    else:
        X_init_norm = X_train_norm[init_indices]
        hmwp_init_unnorm = hmwp_train_unnorm[init_indices]

    if not df_val.empty:
        df_val_norm_inputs = normalize_inputs(df_val, scalers)
        t_val_norm = jnp.array(df_val_norm_inputs['Time'].values[:, None])
        T_val_norm = jnp.array(df_val_norm_inputs['Temperature_K'].values[:, None])
        X_val_norm = jnp.hstack([t_val_norm, T_val_norm])
        hmwp_val_unnorm = jnp.array(df_val['HMWP'].values[:, None])
        T_val_K_unnorm = jnp.array(df_val['Temperature_K'].values[:, None])
    else:
        X_val_norm = jnp.empty((0, X_train_norm.shape[1]))
        hmwp_val_unnorm = jnp.empty((0, hmwp_train_unnorm.shape[1]))
        T_val_K_unnorm = jnp.empty((0, 1))
        print("No validation data found.")

    return (X_train_norm, hmwp_train_unnorm, T_train_K_unnorm,
            X_init_norm, hmwp_init_unnorm,
            X_val_norm, hmwp_val_unnorm, T_val_K_unnorm,
            scalers, original_data)


# --- PINN Model Definition ---
class PINN(eqx.Module):
    """Physics-Informed Neural Network model. Predicts UNNORMALIZED HMWP."""
    mlp: eqx.nn.MLP
    # Trainable physics parameters
    log_A: jax.Array
    Ea: jax.Array
    log_HMWP_max: jax.Array
    # --- New SB parameters ---
    n_param: jax.Array # Use internal param name to avoid conflict with 'n'
    m_param: jax.Array # Use internal param name

    def __init__(self, key: jax.random.PRNGKey, scalers: Dict, in_size: int = 2, out_size: int = 1, width_size: int = WIDTH_SIZE, depth: int = DEPTH):
        mlp_key, param_key_A, param_key_Ea, param_key_max, param_key_n, param_key_m = jax.random.split(key, 6)
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=out_size, width_size=width_size, depth=depth,
            activation=jax.nn.tanh,
            key=mlp_key
        )
        # Initialize Arrhenius params
        self.log_A = jax.random.uniform(param_key_A, (1,), minval=jnp.log(1e-2), maxval=jnp.log(1e8))
        self.Ea = jax.random.uniform(param_key_Ea, (1,), minval=20.0, maxval=150.0)

        # Initialize HMWP_max
        observed_max = scalers['HMWP_observed']['max']
        initial_hmwp_max = np.maximum(5.0, observed_max * 1.5)
        self.log_HMWP_max = jnp.log(jnp.array([initial_hmwp_max]))
        print(f"Initial guess for HMWP_max: {jnp.exp(self.log_HMWP_max).item():.2f}")

        # Initialize n and m parameters (using internal names)
        # Initialize around 1.0, ensure positivity via softplus later
        self.n_param = jax.random.uniform(param_key_n, (1,), minval=0.5, maxval=1.5)
        self.m_param = jax.random.uniform(param_key_m, (1,), minval=0.0, maxval=1.0) # m often smaller or zero


    def __call__(self, x_norm: jax.Array) -> jax.Array:
        """Forward pass. Input is NORMALIZED [t, T], Output is UNNORMALIZED HMWP."""
        return self.mlp(x_norm)


# --- Physics Residual Calculation (SB Kinetics) ---
def compute_hmwp_unnorm_from_norm_input(model: PINN, x_norm_combined: jax.Array) -> jax.Array:
    pred = model(x_norm_combined)
    return pred.reshape(())

compute_dhwmp_unnorm_dt_norm = jax.grad(compute_hmwp_unnorm_from_norm_input, argnums=1)

def get_physics_residual_sb(model: PINN, x_norm_combined: jax.Array, T_K_unnorm: jax.Array, scalers: Dict) -> jax.Array:
    """
    Calculates the residual for SB kinetics: d(HMWP)/dt - k(T)*HMWP_max*(1-α)^n*α^m = 0
    where α = HMWP / HMWP_max.
    """
    # Calculate dHMWP/dt (unnormalized)
    dhwmp_unnorm_dt_norm_flat = jax.vmap(compute_dhwmp_unnorm_dt_norm, in_axes=(None, 0))(model, x_norm_combined)
    dhwmp_unnorm_dt_norm = dhwmp_unnorm_dt_norm_flat[:, None]
    time_scale = scalers['Time']['max'] - scalers['Time']['min']
    inv_time_scale = jnp.where(time_scale > 1e-9, 1.0 / time_scale, 0.0)
    dhwmp_dt_unnorm = dhwmp_unnorm_dt_norm * inv_time_scale # This is d(HMWP)/dt

    # Calculate k(T)
    A = jnp.exp(model.log_A)
    Ea_positive = jax.nn.softplus(model.Ea)
    k = A * jnp.exp(-Ea_positive / (R * T_K_unnorm))

    # Get HMWP prediction (unnormalized) and ensure non-negative
    hmwp_pred_unnorm = jax.vmap(model)(x_norm_combined)
    hmwp_pred_unnorm_clipped = jnp.maximum(1e-7, hmwp_pred_unnorm) # Ensure slightly positive for alpha calc

    # Get HMWP_max and ensure positive
    HMWP_max = jnp.exp(model.log_HMWP_max)
    HMWP_max_positive = jnp.maximum(1e-6, HMWP_max)

    # Calculate alpha (fractional conversion)
    alpha = hmwp_pred_unnorm_clipped / HMWP_max_positive
    # Clip alpha to avoid issues at 0 and 1 for pow function
    alpha_clipped = jnp.clip(alpha, 1e-7, 1.0 - 1e-7)

    # Get positive n and m parameters
    n_pos = jax.nn.softplus(model.n_param) # Ensure n >= 0
    m_pos = jax.nn.softplus(model.m_param) # Ensure m >= 0

    # Calculate the SB kinetic term: k * (1-α)^n * α^m
    # Note: The original equation is dα/dt = k*(1-α)^n*α^m
    # Since dα/dt = (1/HMWP_max) * dHMWP/dt
    # => dHMWP/dt = HMWP_max * k * (1-α)^n * α^m
    sb_rate = HMWP_max_positive * k * (1.0 - alpha_clipped)**n_pos * alpha_clipped**m_pos

    # Residual: dHMWP/dt - rate = 0
    residual = dhwmp_dt_unnorm - sb_rate
    return residual


# --- Loss Function ---
def loss_fn(model: PINN, x_norm: jax.Array, y_unnorm: jax.Array, T_K_unnorm: jax.Array,
            x_init_norm: jax.Array, y_init_unnorm: jax.Array,
            scalers: Dict, physics_weight: float) -> jax.Array:
    """Calculates the total loss using SB kinetics."""
    y_pred_unnorm = jax.vmap(model)(x_norm)
    y_pred_unnorm_clipped = jnp.maximum(0.0, y_pred_unnorm)
    loss_data = jnp.mean((y_pred_unnorm_clipped - y_unnorm)**2)

    loss_init = 0.0
    if x_init_norm.shape[0] > 0:
        y_init_pred_unnorm = jax.vmap(model)(x_init_norm)
        y_init_pred_unnorm_clipped = jnp.maximum(0.0, y_init_pred_unnorm)
        loss_init = jnp.mean((y_init_pred_unnorm_clipped - y_init_unnorm)**2)

    # Physics Loss (SB Kinetics)
    physics_residual = get_physics_residual_sb(model, x_norm, T_K_unnorm, scalers)
    loss_physics = jnp.mean(physics_residual**2)

    total_loss = loss_data + loss_init + physics_weight * loss_physics
    return total_loss

# --- Training Step ---
@eqx.filter_jit
def make_step(model: PINN, opt_state: optax.OptState, optimizer: optax.GradientTransformation,
              x_norm: jax.Array, y_unnorm: jax.Array, T_K_unnorm: jax.Array,
              x_init_norm: jax.Array, y_init_unnorm: jax.Array,
              scalers: Dict, physics_weight: float):
    """Performs one training step."""
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_norm, y_unnorm, T_K_unnorm, x_init_norm, y_init_unnorm, scalers, physics_weight)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# --- Evaluation Metrics ---
def calculate_metrics(y_true_unnorm: jax.Array, y_pred_unnorm: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculates RMSE and MAE."""
    if y_true_unnorm.shape[0] == 0:
        return jnp.nan, jnp.nan
    y_pred_unnorm_clipped = jnp.maximum(0.0, y_pred_unnorm)
    rmse = jnp.sqrt(jnp.mean((y_true_unnorm - y_pred_unnorm_clipped)**2))
    mae = jnp.mean(jnp.abs(y_true_unnorm - y_pred_unnorm_clipped))
    return rmse, mae

# --- Main Training Loop ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    model_key, data_key = jax.random.split(key)

    try:
        (X_train_norm, hmwp_train_unnorm, T_train_K_unnorm,
         X_init_norm, hmwp_init_unnorm,
         X_val_norm, hmwp_val_unnorm, T_val_K_unnorm,
         scalers, original_data) = load_and_prepare_data(EXCEL_FILE_PATH, ACCELERATED_DATA_MAX_DAYS)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error during data loading: {e}")
        exit()

    model = PINN(model_key, scalers)

    # Optimizer with LR decay and gradient clipping
    lr_schedule = optax.exponential_decay(
        init_value=INITIAL_LEARNING_RATE,
        transition_steps=LR_DECAY_STEPS,
        decay_rate=LR_DECAY_RATE,
        staircase=True
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-5)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("Starting training with SB kinetics model...")
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10000 # Increase patience further for complex model
    loss = jnp.inf

    for epoch in range(EPOCHS):
        model, opt_state, loss = make_step(
            model, opt_state, optimizer,
            X_train_norm, hmwp_train_unnorm, T_train_K_unnorm,
            X_init_norm, hmwp_init_unnorm,
            scalers, PHYSICS_LOSS_WEIGHT
        )

        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"Error: Loss became {'NaN' if jnp.isnan(loss) else 'Infinity'} at epoch {epoch}. Stopping training.")
            break

        if loss < best_loss:
             best_loss = loss
             patience_counter = 0
        else:
             patience_counter += 1

        if patience_counter >= patience_limit:
             print(f"Stopping early at epoch {epoch} due to lack of improvement.")
             break

        if epoch % 1000 == 0 or epoch == EPOCHS - 1:
             y_pred_unnorm_train = jax.vmap(model)(X_train_norm)
             y_pred_unnorm_train_clipped = jnp.maximum(0.0, y_pred_unnorm_train)
             loss_data = jnp.mean((y_pred_unnorm_train_clipped - hmwp_train_unnorm)**2)
             loss_init = 0.0
             if X_init_norm.shape[0] > 0:
                 y_init_pred_unnorm = jax.vmap(model)(X_init_norm)
                 y_init_pred_unnorm_clipped = jnp.maximum(0.0, y_init_pred_unnorm)
                 loss_init = jnp.mean((y_init_pred_unnorm_clipped - hmwp_init_unnorm)**2)

             physics_residual = get_physics_residual_sb(model, X_train_norm, T_train_K_unnorm, scalers)
             loss_physics = jnp.mean(physics_residual**2)

             try:
                 current_step_count = opt_state[1][0].count
             except (IndexError, AttributeError):
                 try: current_step_count = opt_state[0].count
                 except (IndexError, AttributeError): current_step_count = -1

             current_lr = lr_schedule(current_step_count) if current_step_count != -1 else float('nan')
             print(f"Epoch: {epoch}, LR: {current_lr:.2e}, Total Loss: {loss:.4e}, Data Loss: {loss_data:.4e}, Init Loss: {loss_init:.4e}, Physics Loss: {loss_physics:.4e}")
             # Use softplus to report positive n, m values
             n_report = jax.nn.softplus(model.n_param).item()
             m_report = jax.nn.softplus(model.m_param).item()
             print(f"  Params: Ea={jax.nn.softplus(model.Ea).item():.2f}, log_A={model.log_A.item():.2f}, HMWP_max={jnp.exp(model.log_HMWP_max).item():.2f}, n={n_report:.2f}, m={m_report:.2f}")


    print("Training finished.")
    # --- Evaluation and Prediction ---
    if not (jnp.isnan(loss) or jnp.isinf(loss)):
        print("\nLearned Parameters (Final Model):")
        learned_A = jnp.exp(model.log_A).item()
        learned_Ea = jax.nn.softplus(model.Ea).item()
        learned_HMWP_max = jnp.exp(model.log_HMWP_max).item()
        learned_n = jax.nn.softplus(model.n_param).item()
        learned_m = jax.nn.softplus(model.m_param).item()
        print(f"  Activation Energy (Ea): {learned_Ea:.2f} kJ/mol")
        print(f"  Pre-exponential Factor (A): {learned_A:.4e}")
        print(f"  Predicted HMWP_max: {learned_HMWP_max:.3f} %")
        print(f"  Predicted n: {learned_n:.3f}")
        print(f"  Predicted m: {learned_m:.3f}")

        # --- Calculate Metrics ---
        y_pred_train_unnorm = jax.vmap(model)(X_train_norm)
        train_rmse, train_mae = calculate_metrics(hmwp_train_unnorm, y_pred_train_unnorm)
        print(f"\nTraining Set Metrics:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")

        if X_val_norm.shape[0] > 0:
            y_pred_val_unnorm = jax.vmap(model)(X_val_norm)
            val_rmse, val_mae = calculate_metrics(hmwp_val_unnorm, y_pred_val_unnorm)
            print(f"\nValidation Set Metrics:")
            print(f"  RMSE: {val_rmse:.4f}")
            print(f"  MAE:  {val_mae:.4f}")
        else:
             print("\nNo validation data to evaluate.")

        # --- Plotting ---
        try:
             if 'Temperature' in original_data['train'].columns and 'Temperature' in original_data['val'].columns:
                 all_temps_C = sorted(pd.concat([original_data['train'], original_data['val']])['Temperature'].unique())
             else:
                 all_temps_C = sorted(original_data['train']['Temperature'].unique())
        except KeyError:
             all_temps_C = sorted(original_data['train']['Temperature'].unique())

        plt.figure(figsize=(14, 9))
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_temps_C)))

        for i, tempC in enumerate(all_temps_C):
            tempK = tempC + 273.15
            t_pred_days = jnp.linspace(0, PREDICTION_MAX_DAYS, 200)

            time_scale_val = scalers['Time']['max'] - scalers['Time']['min']
            temp_scale_val = scalers['Temperature_K']['max'] - scalers['Temperature_K']['min']
            t_pred_norm = (t_pred_days - scalers['Time']['min']) / time_scale_val if time_scale_val != 0 else jnp.zeros_like(t_pred_days)
            T_pred_norm_scalar = (tempK - scalers['Temperature_K']['min']) / temp_scale_val if temp_scale_val != 0 else 0.0
            T_pred_norm_scalar_jax = jnp.array(T_pred_norm_scalar)
            T_pred_norm_broadcast = jnp.full_like(t_pred_norm, T_pred_norm_scalar_jax)
            X_pred_norm = jnp.stack([t_pred_norm, T_pred_norm_broadcast], axis=-1)

            hmwp_pred_unnorm = jax.vmap(model)(X_pred_norm)
            hmwp_pred_unnorm_plot = jnp.maximum(0.0, hmwp_pred_unnorm) # Clip negatives

            color = colors[i]
            plt.plot(t_pred_days, hmwp_pred_unnorm_plot, label=f'PINN Prediction ({tempC}°C)', color=color, linestyle='-')

            train_data_temp = original_data['train'][original_data['train']['Temperature'] == tempC]
            val_data_temp = original_data['val'][original_data['val']['Temperature'] == tempC]

            if i == 0:
                 if not train_data_temp.empty: plt.scatter(train_data_temp['Time'], train_data_temp['HMWP'], marker='o', facecolors='none', edgecolors=color, label='Train Data')
                 if not val_data_temp.empty: plt.scatter(val_data_temp['Time'], val_data_temp['HMWP'], marker='x', color=color, label='Validation Data')
            else:
                 if not train_data_temp.empty: plt.scatter(train_data_temp['Time'], train_data_temp['HMWP'], marker='o', facecolors='none', edgecolors=color)
                 if not val_data_temp.empty: plt.scatter(val_data_temp['Time'], val_data_temp['HMWP'], marker='x', color=color)

        plt.xlabel("Time (days)")
        plt.ylabel("HMWP (%)")
        plt.title("PINN Prediction (SB Kinetics) vs Actual Data")
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plot_filename = "pinn_stability_prediction_sb_kinetics.png"
        try:
            plt.savefig(plot_filename)
            print(f"\nPrediction plot saved as {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.show()
        print("\nPrediction complete.")
    else:
        print("\nSkipping prediction, plotting and evaluation due to NaN/Inf loss during training.")

