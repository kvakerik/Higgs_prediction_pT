from ModelClass import RegressionModel
from DatasetClass import DatasetPt, DatasetMass
import logging
import os
import itertools
import matplotlib.pyplot as plt
from src.helpers import extract_data
import numpy as np

# === Nastav logger: iba do súboru, nič do stdout ===
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

def main():
    # === Načítaj dáta ===
    erik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*H125*.root"
    patrik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*Ztt*.root"

    logger.info(f"Loading data from: {erik_data}")

    dataset = DatasetPt(file_paths=erik_data)
    dataset.load_data()
    logger.info("Data loaded successfully.")

    # === Grid search ===
    param_grid = {
        'batch_size': [3000],               
        'learning_rate': [1e-3],             
        'epochs': [200],                     
        'n_layers': [4],                    
        'hidden_layer_size': [4096],     
        'dropout_rate': [0],            
        'weight_decay': [0],            
        "n_normalizer_samples": [10000],      
        "optimizer": ["adamw"]               
    }

    iterable = list(itertools.product(*param_grid.values()))
    best_params = None
    best_loss = float('inf')

    for i, params in enumerate(iterable):
        batch_size_val, learning_rate_val, epochs_val, n_layers_val, hidden_size_val, dropout_val, weight_decay_val, n_normalizer_samples_val, adamw_val = params

        logger.info(f"\n{'='*80}")
        logger.info(
            f"[{i+1}/{len(iterable)}] Training with hyperparameters:\n"
            f"  batch_size={batch_size_val}, learning_rate={learning_rate_val}, epochs={epochs_val},\n"
            f"  n_layers={n_layers_val}, hidden_layer_size={hidden_size_val},\n"
            f"  dropout_rate={dropout_val}, weight_decay={weight_decay_val},\n"
            f"  n_normalizer_samples={n_normalizer_samples_val}, optimizer={adamw_val}"
        )

        model = RegressionModel(
            dataset=dataset,
            batch_size=batch_size_val,
            initial_learning_rate=learning_rate_val,
            n_epochs=epochs_val,
            n_layers=n_layers_val,
            hidden_layer_size=hidden_size_val,
            dropout_rate=dropout_val,
            weight_decay=weight_decay_val,
            n_normalizer_samples = n_normalizer_samples_val,
            optimizer=adamw_val,
        )

        model.prepare_dataset()
        model.create_normalizer()
        model.build_model()
        model.train_model()
        model.plot_history()
        model.save()
        # model.load(model_save_path="model_mmc.keras")


        final_loss = model.history.history['val_mean_squared_error'][-1]
        logger.info(f"Final validation MSE: {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params

    logger.info("\n=== Best hyperparameters ===")
    logger.info(
        f"batch_size={best_params[0]}, learning_rate={best_params[1]}, epochs={best_params[2]},\n"
        f"n_layers={best_params[3]}, hidden_layer_size={best_params[4]},\n"
        f"dropout_rate={best_params[5]}, weight_decay={best_params[6]},\n"
        f"n_normalizer_samples={best_params[7]}, optimizer={best_params[8]}"
    )
    logger.info(f"Lowest achieved validation MSE: {best_loss:.6f}")


    
    y_pred = model.model.predict(model.val_batch)
    y_true = np.array(extract_data(dataset.val_dataset.map(lambda x, y: y))).flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=100, histtype='step', label='True Values', density=True)
    plt.hist(y_pred.flatten(), bins=100, histtype='step', label='Predicted Values', density=True)
    plt.xlabel("Output")
    plt.ylabel("Density")
    plt.title("Predicted vs True Distribution on Validation Set")
    plt.legend()
    plt.grid(True)
    plt.savefig("predicted_vs_true_distribution_mmc.png", dpi=300)


    # Finálne echo do job.out
    print("Training completed. Detailed logs can be found in logs/train.log")

if __name__ == "__main__":
    main()
