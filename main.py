from ModelClass import RegressionModel
from DatasetClass import Dataset, DatasetMass
from src.helpers import make_filter_slice
import tensorflow as tf
import threading
import logging 
import sys
sys.path.append("/home/kvake/.local/lib/python3.11/site-packages")
import optuna   # Import Optuna            
sys.path.append("/home/kvake/.local/lib/python3.9/site-packages")  # Adjust if needed
from optuna_dashboard import run_server

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[
        logging.StreamHandler(sys.stdout), 
    ]
)

logger = logging.getLogger(__name__)  # Create a logger instance

def main():

    patrik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*Ztt*.root"  # No changes here
    logger.info(f"Loading data from: {patrik_data}") # Log the data loading path
    dataset = DatasetMass(file_paths=patrik_data, file_name = "data")
    dataset.load_data()
    #dataset.augment_data_phi(n_slices=1000)
    logger.info("Data loaded successfully.")  # Log success

    def objective(trial):

        try: #Added try and except block for handling possible exceptions
            logger.info(f"Starting trial: {trial.number}") # Log the trial number
            model = RegressionModel(
                dataset,
                n_layers             = trial.suggest_int('n_layers', 2, 4),
                hidden_layer_size    = trial.suggest_int('hidden_layer_size', 512, 1024),
                initial_learning_rate= trial.suggest_float('initial_learning_rate', 2e-3, 1e-2, log=True),
                n_epochs             = trial.suggest_int('n_epochs', 10, 20, step=5),
                activation_function  = 'relu',
                batch_size           = trial.suggest_int('batch_size', 1024, 6144,step=512),
                dropout_rate         = trial.suggest_float('dropout_rate', 0.1, 0.5),
                weight_decay         = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            )

            model.prepare_dataset()
            model.create_normalizer()
            model.build_model()
            model.train_model()

            val_loss, _ = model.model.evaluate(model.val_batch, verbose=0)
            logger.info(f"Trial {trial.number} finished. Validation loss: {val_loss}")  # Log validation loss
            return val_loss

        except Exception as e:
            logger.exception(f"Trial {trial.number} failed: {e}")  # Log exceptions with traceback
            raise  # Re-raise the exception to ensure Optuna handles it correctly

    study = optuna.create_study(storage="sqlite:///optuna.db", study_name="Higgs_analysis_2", direction='minimize', load_if_exists=True)
    logger.info("Optuna study created/loaded.")  # Log study creation

    def run_dashboard():
        server = run_server("sqlite:///optuna.db", host="0.0.0.0", port=8080)
        server.run()  # Run dashboard continuously in a separate thread

    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True  # Ensures it stops when the script ends
    dashboard_thread.start()
    logger.info("Optuna dashboard started.")

    def run_optuna():
        study.optimize(objective, n_trials=100, n_jobs=1)

    # Run Optuna in a separate thread so the dashboard can be accessed in real-time
    optuna_thread = threading.Thread(target=run_optuna)
    optuna_thread.start()

    optuna_thread.join()  # Wait for Optuna to finish before printing results
    logger.info("Optuna optimization finished.") # Log completion
    print("Number of finished trials:", len(study.trials))  # This will still go to stdout (job.out)
    print("Best trial:", study.best_trial.params) # This will still go to stdout (job.out)

if __name__ == "__main__":
    main()