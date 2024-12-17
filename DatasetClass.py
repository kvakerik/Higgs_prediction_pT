import os
import tensorflow as tf
import uproot
import glob
import vector
import awkward as ak
import matplotlib.pyplot as plt 
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
# print("Available GPUs:", gpus)

path_sig = "/scratch/ucjf-atlas/njsf164/data_higgs_root/*VBFH*.root"
path_bkg = "/scratch/ucjf-atlas/njsf164/data_higgs_root/*Ztt*.root"

variables_higgs = [  
    "tau_0_p4",
    "tau_1_p4",
    "ditau_deta","ditau_dphi","ditau_dr","ditau_higgspt","ditau_scal_sum_pt",
    "jet_0_p4",
    "jet_1_p4",
    "dijet_p4",
    "met_p4", 
    "n_jets","n_jets_30","n_jets_40","n_electrons","n_muons","n_taus", 
    "boson_0_truth_p4"
]

class DatasetConstructor():
    def __init__(self, batch_size=64, train_fraction=0.8): 
        self.variables_higgs = variables_higgs 
        self.path_sig = path_sig
        self.path_bkg = path_bkg
        self.batch_size = batch_size
        self.train_fraction = train_fraction

    def importFiles(self):
        files_sig = glob.glob(self.path_sig)
        files_bkg = glob.glob(self.path_bkg)
        #print(f"imported signal files: {files_sig}")
        #print("--"*50)
        #print(f"imported background files: {files_bkg}")
        return files_sig, files_bkg

    def buildDataset(self, plot_variables: bool = False, save_dataset: bool = False):
        # Import files
        files_sig, files_bkg = self.importFiles()
        all_files = files_sig + files_bkg  # Combine signal and background

        print("All files ", len(all_files))
        
        arrays = []
        arrays_truth = []
        
        # Convert awkward arrays to TensorFlow tensors with truth vectors
        tensors = []
        truth_vectors = []
        n_events = []
        
        # Process each file individually
        for file in all_files:
            # print("Reading file", file)
            f = uproot.open(file)['NOMINAL']
            data = f.arrays(self.variables_higgs, library="ak")

            arr = []
            arr_truth = []

            for var in variables_higgs:
                if ('p4' in var) and (var != "boson_0_truth_p4"):
                    # Extract the 4-vector pt, eta, phi, mass
                    p4 = vector.zip({
                        'x': data[var]['fP']['fX'], 
                        'y': data[var]['fP']['fY'], 
                        'z': data[var]['fP']['fZ'],
                        't': data[var]['fE']
                    })
                    
                    arr.append(p4.rho)  # pt
                    arr.append(p4.eta)  # eta
                    arr.append(p4.phi)  # phi
                    arr.append(p4.tau)  # mass

                elif (var == "boson_0_truth_p4"):
                    target_p4 = vector.zip({
                        'x': data[var]['fP']['fX'], 
                        'y': data[var]['fP']['fY'], 
                        'z': data[var]['fP']['fZ'],
                        't': data[var]['fE']
                    })
                    
                    arr_truth.append(target_p4.rho)  # pt
                    arr_truth.append(target_p4.eta)  # eta
                    arr_truth.append(target_p4.phi)  # phi
                    arr_truth.append(target_p4.tau)  # mass
                else:
                    arr.append(data[var])
            
            arrays.append(arr)
            arrays_truth.append(arr_truth)

            if plot_variables:
                # Convert components to numpy arrays
                pt = ak.to_numpy(p4.rho)
                eta = ak.to_numpy(p4.eta)
                phi = ak.to_numpy(p4.phi)
                mass = ak.to_numpy(p4.tau)

                # Plot each component separately
                for component, name in zip([pt, eta, phi, mass], ['pt', 'eta', 'phi', 'mass']):
                    if component.size > 0 and np.issubdtype(component.dtype, np.number):
                        plt.figure()
                        plt.hist(component, bins=50, histtype='step', linestyle='-', linewidth=1.5)
                        plt.title(f"Histogram of {var} - {name}")
                        plt.xlabel(name)
                        plt.ylabel("Events")
                        plt.grid(True)
                        plt.savefig(f"plots/variable_{name}_plot.pdf")
                        plt.close()  # Close the figure after saving
                    else:
                        print(f"Skipping plot for {var} - {name} due to non-numeric or empty data.")
            else:
                print("Plot variables is turned off")
                            
            # Convert variables and truth vectors to tensors
            tensors_var = []
            truths_var = []

            for arr_var in arr:
                # Conversion of variables to TensorFlow tensors
                tensor = tf.constant(ak.to_numpy(arr_var), dtype=tf.float32)
                tensors_var.append(tensor)
            
            for truth_var in arr_truth:
                # Conversion of truth to TensorFlow tensors
                truth_tensor = tf.constant(ak.to_numpy(truth_var), dtype=tf.float32)
                truths_var.append(truth_tensor)

            # Stack variables and truth vectors
            tensor_stack = tf.stack(tensors_var, axis=1)  # Variables (e.g., 35 columns)
            truth_stack = tf.stack(truths_var, axis=1)    # Truth vectors (e.g., 4 columns)

            # Add to lists
            tensors.append(tensor_stack)
            truth_vectors.append(truth_stack)

            n_evt = tensor_stack.shape[0]
            n_events.append(n_evt)

            # Output for verification
            print(f"Processed file {file}: {tensor_stack.shape}, {truth_stack.shape}")
 
        # Total events
        total_events = np.sum(n_events)
        print("Total events", total_events)
        #print(f"Length of tensor lists {len(tensors)}")
        
        datasets = []

        for tensor_file, truth_file in zip(tensors, truth_vectors):
            dataset_sample = tf.data.Dataset.from_tensor_slices((tensor_file, truth_file))
            datasets.append(dataset_sample)
        print("Datasets_len", len(datasets))
        # print(datasets)
        
        train_datasets = []
        val_datasets = []
        train_size_dataset = []
        val_size_dataset = []

        for dataset, dataset_size in zip(datasets, n_events):
            # Determine datasets split sizes
            train_size = int(self.train_fraction * dataset_size)
            val_size = dataset_size - train_size
            
            if val_size > 0 and train_size >= 1:
                # Split datasets into train and validation data
                train_dataset = dataset.take(train_size)
                val_dataset = dataset.skip(train_size)

                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)

                train_size_dataset.append(train_size)
                val_size_dataset.append(val_size)
            else:
                print(f"Skipping file with dataset size: {dataset_size}")

            print(f"Dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}")

        print("Number of training datasets:", len(train_datasets))
        print("Number of validation datasets:", len(val_datasets))

        weights_list_train = []
        weights_list_val = []

        for train_dataset, val_dataset, train_size, val_size in zip(train_datasets, val_datasets, train_size_dataset, val_size_dataset):
            try:
                # Note: tf.data.Dataset does not support len(), so we use train_size and val_size directly
                weights_train = train_size / train_size  # This will always be 1.0
                weights_val = val_size / val_size      # This will always be 1.0
                weights_list_train.append(weights_train)
                weights_list_val.append(weights_val)
            except ZeroDivisionError:
                print("Skipping one event file due to zero division")

        print("Finished weighting")
        print("weights_list_train_len", len(weights_list_train))
        print("weights_list_val_len", len(weights_list_val))

        # Combine all train and validation datasets
        train_dataset = train_datasets[0]
        for ds in train_datasets[1:]:
            train_dataset = train_dataset.concatenate(ds)

        val_dataset = val_datasets[0]
        for ds in val_datasets[1:]:
            val_dataset = val_dataset.concatenate(ds)

        print("Dataset Successfully imported")
        print(type(val_dataset))
        for x in val_dataset.take(1):
            print(x)    

        if save_dataset:
            os.makedirs("data", exist_ok=True)  # Ensure 'data' directory exists
            val_dataset.save("data/val_dataset")
            train_dataset.save("data/train_dataset")

            train_events = int(self.train_fraction * total_events)
            val_events = int((1 - self.train_fraction) * total_events)

            # Save train_events and val_events to a text file
            output_file = "data/event_counts.txt"
            with open(output_file, "w") as f:
                f.write(f"{train_events}\n")
                f.write(f"{val_events}\n")
            print(f"Saved event counts to {output_file}")

        return val_dataset, train_dataset, val_events, train_events 

if __name__ == "__main__":
    datasetConstructor = DatasetConstructor()
    val_dataset, train_dataset, val_events, train_events = datasetConstructor.buildDataset(plot_variables=False, save_dataset=True)
    