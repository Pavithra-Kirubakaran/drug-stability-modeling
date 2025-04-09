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
INITIAL_LEARNING_RATE = 1e-3 # Initial LR for decay schedule
EPOCHS = 40000 # Increase epochs further
PHYSICS_LOSS_WEIGHT = 1e-3 # Try increasing physics weight
WIDTH_SIZE = 64
DEPTH = 4
LR_DECAY_RATE = 0.9 # Learning rate decay rate
LR_DECAY_STEPS = 1000 # Steps for LR decay
GRAD_CLIP_NORM = 1.0 # Global norm for gradient clipping
# --- End Hyperparameters ---
R = 8.314e-3 # Gas constant in kJ/(mol*K)

# --- Data Loading and Preparation ---
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
    log_A: jax.Array
    Ea: jax.Array
    log_HMWP_max: jax.Array

    def __init__(self, key: jax.random.PRNGKey, scalers: Dict, in_size: int = 2, out_size: int = 1, width_size: int = WIDTH_SIZE, depth: int = DEPTH):
        mlp_key, param_key1, param_key2, param_key3 = jax.random.split(key, 4)
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=out_size, width_size=width_size, depth=depth,
            activation=jax.nn.tanh, # Keep tanh for internal layers
            # No final activation here
            key=mlp_key
        )
        self.log_A = jax.random.uniform(param_key1, (1,), minval=jnp.log(1e-2), maxval=jnp.log(1e8))
        self.Ea = jax.random.uniform(param_key2, (1,), minval=20.0, maxval=150.0)

        # Initialize log_HMWP_max higher and less restrictively
        observed_max = scalers['HMWP_observed']['max']
        # Ensure initial guess is reasonably high, e.g., at least 5% or 1.5x observed max
        initial_hmwp_max = np.maximum(5.0, observed_max * 1.5)
        self.log_HMWP_max = jnp.log(jnp.array([initial_hmwp_max]))
        print(f"Initial guess for HMWP_max: {jnp.exp(self.log_HMWP_max).item():.2f}")


    def __call__(self, x_norm: jax.Array) -> jax.Array:
        """Forward pass. Input is NORMALIZED [t, T], Output is UNNORMALIZED HMWP."""
        # Remove final softplus, model predicts raw value
        return self.mlp(x_norm)


# --- Physics Residual Calculation (First Order Kinetics) ---
def compute_hmwp_unnorm_from_norm_input(model: PINN, x_norm_combined: jax.Array) -> jax.Array:
    pred = model(x_norm_combined)
    return pred.reshape(())

compute_dhwmp_unnorm_dt_norm = jax.grad(compute_hmwp_unnorm_from_norm_input, argnums=1)

def get_physics_residual_first_order(model: PINN, x_norm_combined: jax.Array, T_K_unnorm: jax.Array, scalers: Dict) -> jax.Array:
    dhwmp_unnorm_dt_norm_flat = jax.vmap(compute_dhwmp_unnorm_dt_norm, in_axes=(None, 0))(model, x_norm_combined)
    dhwmp_unnorm_dt_norm = dhwmp_unnorm_dt_norm_flat[:, None]

    time_scale = scalers['Time']['max'] - scalers['Time']['min']
    inv_time_scale = jnp.where(time_scale > 1e-9, 1.0 / time_scale, 0.0)
    dhwmp_dt_unnorm = dhwmp_unnorm_dt_norm * inv_time_scale

    A = jnp.exp(model.log_A)
    Ea_positive = jax.nn.softplus(model.Ea) # Keep Ea positive
    k = A * jnp.exp(-Ea_positive / (R * T_K_unnorm))

    hmwp_pred_unnorm = jax.vmap(model)(x_norm_combined)
    # Clip negative predictions if they occur after removing softplus
    hmwp_pred_unnorm_clipped = jnp.maximum(0.0, hmwp_pred_unnorm)

    HMWP_max = jnp.exp(model.log_HMWP_max)
    # Ensure HMWP_max is also positive (should be due to exp)
    HMWP_max_positive = jnp.maximum(1e-6, HMWP_max) # Ensure slightly positive

    # Ensure HMWP_pred doesn't exceed HMWP_max
    hmwp_pred_safe = jnp.minimum(hmwp_pred_unnorm_clipped, HMWP_max_positive * 0.9999)

    driving_force = k * (HMWP_max_positive - hmwp_pred_safe)
    # Ensure driving force is non-negative
    driving_force_safe = jnp.maximum(0.0, driving_force)

    residual = dhwmp_dt_unnorm - driving_force_safe
    return residual


# --- Loss Function ---
def loss_fn(model: PINN, x_norm: jax.Array, y_unnorm: jax.Array, T_K_unnorm: jax.Array,
            x_init_norm: jax.Array, y_init_unnorm: jax.Array,
            scalers: Dict, physics_weight: float) -> jax.Array:
    y_pred_unnorm = jax.vmap(model)(x_norm)
    # Clip negative predictions before calculating loss
    y_pred_unnorm_clipped = jnp.maximum(0.0, y_pred_unnorm)
    loss_data = jnp.mean((y_pred_unnorm_clipped - y_unnorm)**2)

    loss_init = 0.0
    if x_init_norm.shape[0] > 0:
        y_init_pred_unnorm = jax.vmap(model)(x_init_norm)
        y_init_pred_unnorm_clipped = jnp.maximum(0.0, y_init_pred_unnorm)
        loss_init = jnp.mean((y_init_pred_unnorm_clipped - y_init_unnorm)**2)

    physics_residual = get_physics_residual_first_order(model, x_norm, T_K_unnorm, scalers)
    loss_physics = jnp.mean(physics_residual**2)

    total_loss = loss_data + loss_init + physics_weight * loss_physics
    return total_loss

# --- Training Step ---
@eqx.filter_jit
def make_step(model: PINN, opt_state: optax.OptState, optimizer: optax.GradientTransformation,
              x_norm: jax.Array, y_unnorm: jax.Array, T_K_unnorm: jax.Array,
              x_init_norm: jax.Array, y_init_unnorm: jax.Array,
              scalers: Dict, physics_weight: float):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_norm, y_unnorm, T_K_unnorm, x_init_norm, y_init_unnorm, scalers, physics_weight)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# --- Evaluation Metrics ---
def calculate_metrics(y_true_unnorm: jax.Array, y_pred_unnorm: jax.Array) -> Tuple[jax.Array, jax.Array]:
    if y_true_unnorm.shape[0] == 0:
        return jnp.nan, jnp.nan
    # Clip negative predictions before evaluation
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
        staircase=True # Apply decay at discrete steps
    )
    # Chain AdamW, gradient clipping, and LR schedule
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-5)
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("Starting training with first-order kinetics model (Optimized v3)...")
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 8000 # Increase patience slightly
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
            print("Consider adjusting hyperparameters (LR, physics weight), clipping, or checking data.")
            break

        # Simple loss-based checkpointing (save best model state)
        if loss < best_loss:
             best_loss = loss
             patience_counter = 0
             # Could save the best model here if needed: best_model = model
        else:
             patience_counter += 1

        if patience_counter >= patience_limit:
             print(f"Stopping early at epoch {epoch} due to lack of improvement.")
             break

        if epoch % 1000 == 0 or epoch == EPOCHS - 1:
             # Calculate losses with clipped predictions for reporting consistency
             y_pred_unnorm_train = jax.vmap(model)(X_train_norm)
             y_pred_unnorm_train_clipped = jnp.maximum(0.0, y_pred_unnorm_train)
             loss_data = jnp.mean((y_pred_unnorm_train_clipped - hmwp_train_unnorm)**2)

             loss_init = 0.0
             if X_init_norm.shape[0] > 0:
                 y_init_pred_unnorm = jax.vmap(model)(X_init_norm)
                 y_init_pred_unnorm_clipped = jnp.maximum(0.0, y_init_pred_unnorm)
                 loss_init = jnp.mean((y_init_pred_unnorm_clipped - hmwp_init_unnorm)**2)

             physics_residual = get_physics_residual_first_order(model, X_train_norm, T_train_K_unnorm, scalers)
             loss_physics = jnp.mean(physics_residual**2)

             # --- FIXED: Correct access to step count in opt_state ---
             # Assuming opt_state structure: (ClipState, AdamWState)
             # AdamWState structure: (ScaleByAdamState, AddDecayedWeightsState)
             # Count is in ScaleByAdamState
             try:
                 # Access count assuming nested state: (ClipState, (ScaleByAdamState, AddDecayedWeightsState))
                 current_step_count = opt_state[1][0].count
             except (IndexError, AttributeError):
                 # Fallback or alternative access if structure differs (e.g., if only AdamW was used)
                 try:
                    current_step_count = opt_state[0].count # If AdamW is the first state
                 except (IndexError, AttributeError):
                    print("Warning: Could not determine step count for LR logging.")
                    current_step_count = -1 # Placeholder

             current_lr = lr_schedule(current_step_count) if current_step_count != -1 else float('nan')
             print(f"Epoch: {epoch}, LR: {current_lr:.2e}, Total Loss: {loss:.4e}, Data Loss: {loss_data:.4e}, Init Loss: {loss_init:.4e}, Physics Loss: {loss_physics:.4e}")
             # --- End Fix ---
             print(f"  Current Ea: {jax.nn.softplus(model.Ea).item():.2f}, log_A: {model.log_A.item():.2f}, HMWP_max: {jnp.exp(model.log_HMWP_max).item():.2f}")


    print("Training finished.")
    # --- Evaluation and Prediction ---
    # Check if training stopped early due to NaN/Inf loss
    if not (jnp.isnan(loss) or jnp.isinf(loss)):
        # If checkpointing was used, load the best model here: model = best_model
        print("\nLearned Parameters (Final Model):")
        learned_A = jnp.exp(model.log_A).item()
        learned_Ea = jax.nn.softplus(model.Ea).item()
        learned_HMWP_max = jnp.exp(model.log_HMWP_max).item()
        print(f"  Activation Energy (Ea): {learned_Ea:.2f} kJ/mol")
        print(f"  Pre-exponential Factor (A): {learned_A:.4e}")
        print(f"  Predicted HMWP_max: {learned_HMWP_max:.3f} %")

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
            # Clip negative predictions for plotting
            hmwp_pred_unnorm_plot = jnp.maximum(0.0, hmwp_pred_unnorm)

            color = colors[i]
            plt.plot(t_pred_days, hmwp_pred_unnorm_plot, label=f'PINN Prediction ({tempC}°C)', color=color, linestyle='-')

            train_data_temp = original_data['train'][original_data['train']['Temperature'] == tempC]
            val_data_temp = original_data['val'][original_data['val']['Temperature'] == tempC]

            if i == 0:
                 if not train_data_temp.empty:
                     plt.scatter(train_data_temp['Time'], train_data_temp['HMWP'], marker='o', facecolors='none', edgecolors=color, label='Train Data')
                 if not val_data_temp.empty:
                     plt.scatter(val_data_temp['Time'], val_data_temp['HMWP'], marker='x', color=color, label='Validation Data')
            else:
                 if not train_data_temp.empty:
                     plt.scatter(train_data_temp['Time'], train_data_temp['HMWP'], marker='o', facecolors='none', edgecolors=color)
                 if not val_data_temp.empty:
                     plt.scatter(val_data_temp['Time'], val_data_temp['HMWP'], marker='x', color=color)

        plt.xlabel("Time (days)")
        plt.ylabel("HMWP (%)")
        plt.title("PINN Prediction (First Order Kinetics - Optimized v3) vs Actual Data")
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plot_filename = "pinn_stability_prediction_first_order_v3.png"
        try:
            plt.savefig(plot_filename)
            print(f"\nPrediction plot saved as {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.show()
        print("\nPrediction complete.")
    else:
        print("\nSkipping prediction, plotting and evaluation due to NaN/Inf loss during training.")

