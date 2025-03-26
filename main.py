from ModelClass import RegressionModel
from DatasetClass import DatasetPt, DatasetMass
import logging
import os
import itertools

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

    logger.info(f"Loading data from: {patrik_data}")

    dataset = DatasetMass(file_paths=patrik_data, file_name="data")
    dataset.load_data()
    logger.info("Data loaded successfully.")

    # === Grid search ===
    param_grid = {
        'batch_size': [3000, 2500],
        'learning_rate': [3e-3, 6e-3],
        'epochs': [30, 40],
        'n_layers': [2,3,4],
        'hidden_layer_size': [1024, 1536],
        'dropout_rate': [0.2, 0.3],
        'weight_decay': [1e-5],
        "n_normalizer_samples": [20, 30, 40]
    }

    iterable = list(itertools.product(*param_grid.values()))
    best_params = None
    best_loss = float('inf')

    for i, params in enumerate(iterable):
        batch_size_val, learning_rate_val, epochs_val, n_layers_val, hidden_size_val, dropout_val, weight_decay_val, n_normalizer_samples_val = params

        logger.info(f"\n{'='*80}")
        logger.info(
            f"[{i+1}/{len(iterable)}] Trénujeme s hyperparametrami:\n"
            f"  batch_size={batch_size_val}, learning_rate={learning_rate_val}, epochs={epochs_val},\n"
            f"  n_layers={n_layers_val}, hidden_layer_size={hidden_size_val},\n"
            f"  dropout_rate={dropout_val}, weight_decay={weight_decay_val}"
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
            n_normalizer_samples = n_normalizer_samples_val
        )

        model.prepare_dataset()
        model.create_normalizer()
        model.build_model()
        model.train_model()
        model.plot_history()

        final_loss = model.history.history['loss'][-1]
        logger.info(f"Final training loss: {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params

    logger.info("\n=== Najlepšie hyperparametre ===")
    logger.info(
        f"batch_size={best_params[0]}, learning_rate={best_params[1]}, epochs={best_params[2]},\n"
        f"n_layers={best_params[3]}, hidden_layer_size={best_params[4]},\n"
        f"dropout_rate={best_params[5]}, weight_decay={best_params[6]}"
    )
    logger.info(f"Najnižšia dosažená tréningová loss: {best_loss:.6f}")

    # Finálne echo do job.out
    print("Tréning dokončený detailné logy nájdeš v logs/train.log")

if __name__ == "__main__":
    main()
