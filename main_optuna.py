import sys
sys.path.append("/home/kvake/.local/lib/python3.9/site-packages")     # optuna
sys.path.append("/usr/local/lib64/python3.9/site-packages")           # sqlalchemy
sys.path.append("/usr/local/lib/python3.9/site-packages")             # alembic

import sqlalchemy
import logging
import os
import threading
import optuna
from optuna_dashboard import run_server
from ModelClass import RegressionModel
from DatasetClass import DatasetPt
from src.helpers import EpochLogger

# === Logger setup ===
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

# === Objective function for Optuna ===
def objective(trial):
    model = RegressionModel(
        dataset,
        n_layers=trial.suggest_int('n_layers', 3, 6),
        hidden_layer_size=trial.suggest_int('hidden_layer_size',2048, 4096, step=512),
        initial_learning_rate=trial.suggest_float('initial_learning_rate', 1e-2, 1e-1, log=True),
        n_epochs=80,
        activation_function  = 'relu',
        batch_size=trial.suggest_int('batch_size', 5000, 9000, step=1000),
        dropout_rate=0.2,
        weight_decay= 1e-5,
        optimizer='adamw',
        n_normalizer_samples = trial.suggest_int('n_normalizer_samples',10000,100000, step=10000)
    )
    
    model.prepare_dataset()
    model.create_normalizer()
    model.build_model()
    model.train_model()

    evaluation_results = model.model.evaluate(model.dev_batch, verbose=0)
    val_loss = evaluation_results[0] if isinstance(evaluation_results, list) else evaluation_results
    logger.info(f"Trial done. Validation MSE: {val_loss:.6f} | Params: {trial.params}")
    return val_loss

# === Spustenie Optuna dashboardu ===
def run_dashboard():
    server = run_server("sqlite:///optuna.db", host="0.0.0.0", port=8080)
    server.run()

dashboard_thread = threading.Thread(target=run_dashboard)
dashboard_thread.daemon = True

# === Hlavná funkcia ===
def main():
    global dataset
    logger.info("Loading dataset")
    erik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*H125*.root"
    patrik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*Ztt*.root"
    dataset = DatasetPt(file_paths=erik_data)
    dataset.load_data()
    logger.info("Dataset loaded successfully.")
    print("Train dataset size:", len(dataset.train_dataset))

    dashboard_thread.start()

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="Higgs_analysis_optuna_vyssia_kapacita",
        direction="minimize",
        load_if_exists=True
    )

    def run_optuna():
        study.optimize(objective, n_trials=100, n_jobs=1)

    optuna_thread = threading.Thread(target=run_optuna)
    optuna_thread.start()
    optuna_thread.join()

    logger.info("\n=== Best hyperparameters ===")
    logger.info(study.best_trial.params)
    logger.info(f"Lowest achieved validation MSE: {study.best_value:.6f}")
    print("\nTraining completed. Detailed logs can be found in logs/train.log")
    
if __name__ == "__main__":
    main()

