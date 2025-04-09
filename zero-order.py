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
# Construct the path relative to the script location or use an absolute path
# Assuming 'Data' folder is in the same directory as the script
script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
EXCEL_FILE_PATH = os.path.join(script_dir, 'Data', 'stability_data.xlsx')
# EXCEL_FILE_PATH = 'Data/stability_data.xlsx' # Original path, might work depending on execution context

ACCELERATED_DATA_MAX_DAYS = 97
PREDICTION_MAX_DAYS = 3 * 365 # Extrapolate to 3 years
SEED = 123
LEARNING_RATE = 1e-3
EPOCHS = 20000 # Adjust as needed, PINNs often require many epochs
PHYSICS_LOSS_WEIGHT = 1e-2 # Lambda to balance data and physics loss
R = 8.314e-3 # Gas constant in kJ/(mol*K)

# --- Data Loading and Preparation ---
def load_and_prepare_data(file_path: str, max_train_days: int) -> Tuple:
    """Loads data from Excel, splits, normalizes, and converts to JAX arrays."""
    # Check if the file exists before attempting to load
    if not os.path.exists(file_path):
         print(f"Error: Excel file not found at the specified path: {os.path.abspath(file_path)}")
         print("Please ensure the 'Data' folder exists in the same directory as the script and contains 'stability_data.xlsx'")
         print("Or, adjust the EXCEL_FILE_PATH variable in the script.")
         # Provide dummy data path for demonstration if needed, or raise error
         raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from {file_path}")
    except Exception as e:
        print(f"Error loading or reading Excel file {file_path}: {e}")
        raise

    # Ensure required columns exist
    required_cols = ['Temperature', 'Time', 'HMWP']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    # Drop rows with any NaN values in essential columns
    df.dropna(subset=required_cols, inplace=True)
    if df.empty:
        raise ValueError("Dataframe is empty after dropping NaN values. Check your Excel file.")


    # Convert Temperature to Kelvin
    df['Temperature_K'] = df['Temperature'] + 273.15

    # Separate training (accelerated) and validation (long-term) data
    df_train = df[df['Time'] <= max_train_days].copy()
    df_val = df[df['Time'] > max_train_days].copy()

    # Store original data for plotting (including PM if exists)
    original_data = {'train': df_train.copy(), 'val': df_val.copy()}

    # Check if training data is empty
    if df_train.empty:
        raise ValueError(f"No training data found with Time <= {max_train_days} days.")

    # Normalize features (Time, Temperature_K) and target (HMWP)
    # Using min-max scaling based on the *training* data range
    scalers = {
        'Time': {'min': df_train['Time'].min(), 'max': df_train['Time'].max()},
        'Temperature_K': {'min': df_train['Temperature_K'].min(), 'max': df_train['Temperature_K'].max()},
        'HMWP': {'min': df_train['HMWP'].min(), 'max': df_train['HMWP'].max()}
    }
    print("Scalers calculated based on training data:", scalers)


    def normalize(df, scalers):
        df_norm = df.copy()
        for col, scale in scalers.items():
            min_val, max_val = scale['min'], scale['max']
            # Avoid division by zero if min == max (constant value)
            if max_val - min_val == 0:
                 df_norm[col] = 0.0 # Assign a consistent normalized value (e.g., 0)
            else:
                 df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        return df_norm

    df_train_norm = normalize(df_train, scalers)

    # Prepare JAX arrays for training
    t_train = jnp.array(df_train_norm['Time'].values[:, None])
    T_train_norm = jnp.array(df_train_norm['Temperature_K'].values[:, None]) # Renamed for clarity
    hmwp_train = jnp.array(df_train_norm['HMWP'].values[:, None])
    X_train = jnp.hstack([t_train, T_train_norm]) # Combine inputs

    # Also keep unnormalized Kelvin temperature for Arrhenius calculation
    T_train_K_unnorm = jnp.array(df_train['Temperature_K'].values[:, None])

    # Prepare initial condition data (t=0)
    # Find indices where *unnormalized* time is 0
    init_indices = np.where(df_train['Time'].values == 0.0)[0]
    if len(init_indices) == 0:
         print("Warning: No data points found at Time = 0 for initial condition loss.")
         # Create empty arrays with correct shape to avoid errors downstream
         X_init = jnp.empty((0, X_train.shape[1]))
         hmwp_init = jnp.empty((0, hmwp_train.shape[1]))
         T_init_K_unnorm = jnp.empty((0, T_train_K_unnorm.shape[1]))
    else:
        X_init = X_train[init_indices]
        hmwp_init = hmwp_train[init_indices]
        T_init_K_unnorm = T_train_K_unnorm[init_indices]


    return X_train, hmwp_train, T_train_K_unnorm, X_init, hmwp_init, T_init_K_unnorm, scalers, original_data


# --- PINN Model Definition ---
class PINN(eqx.Module):
    """Physics-Informed Neural Network model."""
    mlp: eqx.nn.MLP
    # Trainable physics parameters (logA and Ea)
    log_A: jax.Array # Log of pre-exponential factor
    Ea: jax.Array    # Activation energy

    def __init__(self, key: jax.random.PRNGKey, in_size: int = 2, out_size: int = 1, width_size: int = 32, depth: int = 4):
        """Initializes the MLP and physics parameters."""
        mlp_key, param_key1, param_key2 = jax.random.split(key, 3)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh, # Tanh activation often works well
            key=mlp_key
        )
        # Initialize physics parameters - scaling might be needed depending on expected magnitude
        # Adjusted initialization ranges based on typical chemical kinetics
        self.log_A = jax.random.uniform(param_key1, (1,), minval=jnp.log(1e4), maxval=jnp.log(1e15)) # Broader range for log(A) [units depend on order, here effectively 1/days]
        self.Ea = jax.random.uniform(param_key2, (1,), minval=20.0, maxval=150.0) # Typical Ea range (kJ/mol)


    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP."""
        # x contains [normalized_time, normalized_temp_K]
        return self.mlp(x)

# --- Physics Residual Calculation (Corrected) ---

# Define the function that computes HMWP for a single input point
# Takes the *combined* normalized input [t_norm, T_norm]
def compute_hmwp_from_combined_input(model: PINN, x_combined: jax.Array) -> jax.Array:
    """Computes the NN output HMWP for a single combined input x = [t_norm, T_norm]."""
    # Ensure the output is scalar for grad calculation
    pred = model(x_combined)
    return pred.reshape(()) # Return scalar

# Get the gradient function for HMWP w.r.t. the *first element* (t_norm) of the combined input x_combined
# Note: argnums=1 because model is the 0th arg to compute_hmwp_from_combined_input
compute_hmwp_dt_norm = jax.grad(compute_hmwp_from_combined_input, argnums=1)

def get_physics_residual(model: PINN, x_combined: jax.Array, T_K_unnorm: jax.Array, scalers: Dict) -> jax.Array:
    """
    Calculates the residual of the governing PDE: d(HMWP)/dt - k(T) = 0.
    Args:
        model: The PINN model.
        x_combined: Combined normalized inputs [t_norm, T_norm] with shape (N, 2).
        T_K_unnorm: Unnormalized temperatures in Kelvin with shape (N, 1).
        scalers: Dictionary containing min/max values for normalization.
    Returns:
        The physics residual for each input point, shape (N, 1).
    """
    # Calculate d(HMWP_norm)/d(t_norm) using automatic differentiation and vmap
    # vmap over the function that calculates the gradient for a single point
    # Pass model as static (None), vmap over x_combined (axis 0)
    hmwp_t_norm_flat = jax.vmap(compute_hmwp_dt_norm, in_axes=(None, 0))(model, x_combined)
    # Reshape to (N, 1) to match other terms
    hmwp_t_norm = hmwp_t_norm_flat[:, None]

    # Calculate reaction rate k using Arrhenius equation with current parameters
    A = jnp.exp(model.log_A)
    # Ensure Ea is positive as expected physically
    Ea_positive = jax.nn.softplus(model.Ea) # Or simply jnp.abs(model.Ea) if preferred
    k = A * jnp.exp(-Ea_positive / (R * T_K_unnorm)) # Use unnormalized Kelvin temp

    # The physics equation is d(HMWP)/dt = k
    # We need to relate d(HMWP_norm)/d(t_norm) to k using the chain rule and normalization factors.
    # d(HMWP)/dt = [d(HMWP_norm)/d(t_norm)] * [(HMWP_max - HMWP_min) / (t_max - t_min)]
    hmwp_scale = scalers['HMWP']['max'] - scalers['HMWP']['min']
    time_scale = scalers['Time']['max'] - scalers['Time']['min']

    # --- JIT-compatible conditional logic using jnp.where ---
    # Condition: both scales are non-zero (use floating point comparison)
    # Add a small epsilon to avoid issues with exact zero comparison if necessary,
    # but direct comparison should be okay for values derived from data.
    condition = (time_scale > 1e-9) & (hmwp_scale > 1e-9) # Check if scales are effectively non-zero

    # If true (condition is met), calculate scale_factor, otherwise use 0.0
    # Ensure division happens only when time_scale is non-zero
    safe_time_scale = jnp.where(time_scale > 1e-9, time_scale, 1.0) # Avoid division by zero
    calculated_scale_factor = hmwp_scale / safe_time_scale
    scale_factor = jnp.where(condition, calculated_scale_factor, 0.0)
    # --- End of jnp.where logic ---

    # Calculate the residual: d(HMWP)/dt - k = 0
    # residual = (hmwp_t_norm * scale_factor) - k
    # Ensure hmwp_t_norm has shape (N, 1) before scaling
    residual = (hmwp_t_norm * scale_factor) - k

    return residual


# --- Loss Function ---
def loss_fn(model: PINN, x: jax.Array, y: jax.Array, T_K_unnorm: jax.Array,
            x_init: jax.Array, y_init: jax.Array, T_init_K_unnorm: jax.Array, # T_init_K_unnorm is not strictly needed here but kept for consistency
            scalers: Dict, physics_weight: float) -> jax.Array:
    """Calculates the total loss (Data MSE + Initial Cond MSE + Physics Residual MSE)."""
    # Data Loss (MSE on training points)
    y_pred = jax.vmap(model)(x)
    loss_data = jnp.mean((y_pred - y)**2)

    # Initial Condition Loss (MSE on t=0 points)
    # Handle case where there are no initial points
    if x_init.shape[0] > 0:
        y_init_pred = jax.vmap(model)(x_init)
        loss_init = jnp.mean((y_init_pred - y_init)**2)
    else:
        loss_init = 0.0 # No initial points, no loss contribution

    # Physics Loss (MSE on PDE residual at training points)
    physics_residual = get_physics_residual(model, x, T_K_unnorm, scalers)
    loss_physics = jnp.mean(physics_residual**2)

    # Total Loss
    total_loss = loss_data + loss_init + physics_weight * loss_physics

    # Add regularization for Ea to keep it physically reasonable (optional)
    # e.g., penalize very small or very large Ea values if needed
    # reg_loss = 1e-4 * (jax.nn.relu(-model.Ea) + jax.nn.relu(model.Ea - 200)) # Penalize Ea < 0 or Ea > 200
    # total_loss += reg_loss

    return total_loss

# --- Training Step ---
@eqx.filter_jit # Compile the training step for speed
def make_step(model: PINN, opt_state: optax.OptState, optimizer: optax.GradientTransformation,
              x: jax.Array, y: jax.Array, T_K_unnorm: jax.Array,
              x_init: jax.Array, y_init: jax.Array, T_init_K_unnorm: jax.Array,
              scalers: Dict, physics_weight: float):
    """Performs one training step."""
    # Calculate loss and gradients
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, T_K_unnorm, x_init, y_init, T_init_K_unnorm, scalers, physics_weight)

    # Update model parameters
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_val


# --- Main Training Loop ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    model_key, data_key = jax.random.split(key)

    # Load and prepare data
    try:
        X_train, hmwp_train, T_train_K_unnorm, X_init, hmwp_init, T_init_K_unnorm, scalers, original_data = \
            load_and_prepare_data(EXCEL_FILE_PATH, ACCELERATED_DATA_MAX_DAYS)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error during data loading: {e}")
        # Consider exiting gracefully or handling the error appropriately
        exit() # Exit if data loading fails

    # Initialize model and optimizer
    model = PINN(model_key)
    # Use Adam optimizer with weight decay (L2 regularization on MLP weights)
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=1e-5)
    # Filter model to get only trainable parameters for the optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("Starting training...")
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5000 # Stop if loss doesn't improve for 5000 steps
    loss = jnp.inf # Initialize loss to infinity

    for epoch in range(EPOCHS):
        # Ensure inputs to make_step have the correct types/shapes if issues arise
        model, opt_state, loss = make_step(
            model, opt_state, optimizer,
            X_train, hmwp_train, T_train_K_unnorm,
            X_init, hmwp_init, T_init_K_unnorm,
            scalers, PHYSICS_LOSS_WEIGHT
        )

        # Check for NaN loss
        if jnp.isnan(loss):
            print(f"Error: Loss became NaN at epoch {epoch}. Stopping training.")
            print("Check learning rate, physics weight, initialization, or data scaling.")
            break # Stop training if loss is NaN

        # Basic early stopping
        if loss < best_loss:
             best_loss = loss
             patience_counter = 0
        else:
             patience_counter += 1

        if patience_counter >= patience_limit:
             print(f"Stopping early at epoch {epoch} due to lack of improvement.")
             break


        if epoch % 1000 == 0 or epoch == EPOCHS - 1:
             # Also calculate individual loss components for monitoring
             y_pred = jax.vmap(model)(X_train)
             loss_data = jnp.mean((y_pred - hmwp_train)**2)

             if X_init.shape[0] > 0:
                 y_init_pred = jax.vmap(model)(X_init)
                 loss_init = jnp.mean((y_init_pred - hmwp_init)**2)
             else:
                 loss_init = 0.0

             physics_residual = get_physics_residual(model, X_train, T_train_K_unnorm, scalers)
             loss_physics = jnp.mean(physics_residual**2)

             print(f"Epoch: {epoch}, Total Loss: {loss:.4e}, Data Loss: {loss_data:.4e}, Init Loss: {loss_init:.4e}, Physics Loss: {loss_physics:.4e}")
             # Print current physics parameters
             print(f"  Current Ea: {jax.nn.softplus(model.Ea).item():.2f}, Current log_A: {model.log_A.item():.2f}")


    print("Training finished.")

    # --- Prediction and Extrapolation ---
    # Ensure model training didn't stop due to NaN loss before proceeding
    if not jnp.isnan(loss):
        print("\nLearned Parameters:")
        learned_A = jnp.exp(model.log_A).item()
        learned_Ea = jax.nn.softplus(model.Ea).item() # Use softplus ensure positive Ea
        print(f"  Activation Energy (Ea): {learned_Ea:.2f} kJ/mol")
        print(f"  Pre-exponential Factor (A): {learned_A:.4e}")


        # Generate time points for prediction (including extrapolation)
        # Use unique temperatures from the combined original data
        try:
             # Ensure Temperature column exists before concatenation
             if 'Temperature' in original_data['train'].columns and 'Temperature' in original_data['val'].columns:
                 all_temps_C = sorted(pd.concat([original_data['train'], original_data['val']])['Temperature'].unique())
             else:
                 print("Warning: 'Temperature' column missing in train or val data. Using only training temperatures.")
                 all_temps_C = sorted(original_data['train']['Temperature'].unique())
        except KeyError:
             print("Warning: KeyError accessing 'Temperature'. Using only training temperatures.")
             all_temps_C = sorted(original_data['train']['Temperature'].unique())


        prediction_results = {}

        # Denormalize function
        def denormalize(val, col_name, scalers):
            scale = scalers[col_name]
            min_val, max_val = scale['min'], scale['max']
            # Avoid division by zero if min == max
            if max_val - min_val == 0:
                return min_val # Return the constant value
            else:
                # Clip normalized value *before* denormalizing to prevent extreme values
                val_clipped = jnp.clip(val, -0.1, 1.1) # Allow slight overshoot for flexibility
                return val_clipped * (max_val - min_val) + min_val

        plt.figure(figsize=(14, 9)) # Wider figure

        # Use a colormap for different temperatures
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_temps_C)))

        for i, tempC in enumerate(all_temps_C):
            tempK = tempC + 273.15
            t_pred_days = jnp.linspace(0, PREDICTION_MAX_DAYS, 200) # Predict on dense grid

            # Normalize prediction inputs
            time_scale_val = scalers['Time']['max'] - scalers['Time']['min']
            temp_scale_val = scalers['Temperature_K']['max'] - scalers['Temperature_K']['min']

            t_pred_norm = (t_pred_days - scalers['Time']['min']) / time_scale_val if time_scale_val != 0 else jnp.zeros_like(t_pred_days)
            T_pred_norm_scalar = (tempK - scalers['Temperature_K']['min']) / temp_scale_val if temp_scale_val != 0 else 0.0
            # Ensure T_pred_norm_scalar is a JAX array for hstack
            T_pred_norm_scalar_jax = jnp.array(T_pred_norm_scalar)
            T_pred_norm_broadcast = jnp.full_like(t_pred_norm, T_pred_norm_scalar_jax) # Ensure correct shape

            X_pred = jnp.stack([t_pred_norm, T_pred_norm_broadcast], axis=-1)

            # Predict using the trained model
            hmwp_pred_norm = jax.vmap(model)(X_pred)

            # Denormalize predictions
            hmwp_pred = denormalize(hmwp_pred_norm, 'HMWP', scalers)

            prediction_results[tempC] = {'time': t_pred_days, 'hmwp_pred': hmwp_pred}

            # Plotting
            color = colors[i]
            plt.plot(t_pred_days, hmwp_pred, label=f'PINN Prediction ({tempC}°C)', color=color, linestyle='-')

            # Plot original data points for this temperature
            # Handle cases where train/val data might be empty for a specific temp
            train_data_temp = original_data['train'][original_data['train']['Temperature'] == tempC]
            val_data_temp = original_data['val'][original_data['val']['Temperature'] == tempC]

            # Plot training data only once in the legend
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
        plt.title("PINN Prediction of HMWP Stability vs Actual Data")
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)) # Place legend outside plot area
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0) # Ensure HMWP doesn't go below 0
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

        # Save the plot
        plot_filename = "pinn_stability_prediction.png"
        try:
            plt.savefig(plot_filename)
            print(f"\nPrediction plot saved as {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.show() # Also display the plot

        print("\nPrediction complete.")
    else:
        print("\nSkipping prediction and plotting due to NaN loss during training.")

