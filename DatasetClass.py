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

#path_sig = "/scratch/ucjf-atlas/njsf164/data_higgs_root/*VBFH*.root"
#path_bkg = "/scratch/ucjf-atlas/njsf164/data_higgs_root/*Ztt*.root"
class Dataset():
    def __init__(self, **kwargs): 
        default_variables_higgs = [  
                    "tau_0_p4",
                    "tau_1_p4",
                    "ditau_deta","ditau_dphi","ditau_dr","ditau_higgspt","ditau_scal_sum_pt",
                    "jet_0_p4",
                    "jet_1_p4",
                    "dijet_p4",
                    "met_p4", 
                    "n_jets","n_jets_30","n_jets_40","n_electrons","n_muons","n_taus", 
                    ]
        
        default_target_variable = "truth_boson_p4"
        
        self.variables_higgs = kwargs.get('variables_higgs', default_variables_higgs)
        self.target_variable = kwargs.get('target_variable', default_target_variable)
        if self.target_variable not in self.variables_higgs:
            self.variables_higgs.append(self.target_variable)

        self.train_fraction = kwargs.get('train_fraction', 0.8)
        self.file_name = kwargs.get('file_name', "data")
        self.file_paths = kwargs.get('file_paths', "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*.root")
        self.val_dataset = None
        self.train_dataset = None
        self.val_events = None
        self.train_events = None

    def __call__(self):
        self.load_data()

    def importFiles(self): 
        print("Importing Root files")
        files = glob.glob(self.file_paths)
        self.files = files

    def build_dataset(self):
        print("Loading Data")
        self.importFiles()
        all_files = self.files
        print("All files ", len(all_files))
        arrays = []
        arrays_truth = []
        
        # Convert awkward arrays to TensorFlow tensors with truth vectors
        tensors = []
        truth_vectors = []
        n_events = []
        num_processed = 0
        num_skipped = 0        
        # Process each file individually
        for file in all_files[:15]:
            print("Reading file", file)
            f = uproot.open(file)['NOMINAL']
            data = f.arrays(self.variables_higgs, library="ak")
            arr = []
            arr_truth = []
            if len(data) > 0: 
                num_processed += 1
                for var in self.variables_higgs: 
                    #print(var)
                    if ('p4' in var) and (var != self.target_variable):
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

                    elif (var == self.target_variable):
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
            else:
                num_skipped += 1
                print(f"Skipping file {file} due to empty data")
        
        print(num_processed, "files processed")
        print(num_skipped, "files skipped")
        # Total events
        total_events = np.sum(n_events)
        print("Total events", total_events)
        
        datasets = []

        for tensor_file, truth_file in zip(tensors, truth_vectors):
            dataset_sample = tf.data.Dataset.from_tensor_slices((tensor_file, truth_file))
            datasets.append(dataset_sample)
        
        train_datasets = []
        val_datasets = []
        train_size_dataset = []
        val_size_dataset = []

        for dataset, dataset_size in zip(datasets, n_events):
            # Determine datasets split sizes
            train_size = int(self.train_fraction * dataset_size)
            val_size = dataset_size - train_size
            
            if val_size > 0 and train_size > 0:
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

        weights_list_train = [size / sum(train_size_dataset) for size in train_size_dataset]
        weights_list_val = [size / sum(val_size_dataset) for size in val_size_dataset]
        print("Dataset Successfully weighted")
        train_events = int(self.train_fraction * total_events)
        val_events = int(total_events - train_events)

        if len(val_datasets) > 0:
            self.val_dataset = val_datasets[0]
            for ds in val_datasets[1:]:
                self.val_dataset = self.val_dataset.concatenate(ds)

        if len(train_datasets) > 0:
            self.train_dataset = train_datasets[0]
            for ds in train_datasets[1:]:
                self.train_dataset = self.train_dataset.concatenate(ds)

        self.val_events = val_events
        self.train_events = train_events
    
    def save_data(self):  
        print("saving dataset")
        os.makedirs(f"{self.file_name}", exist_ok=True)  # Ensure 'data' directory exists
        val_dataset = self.val_dataset
        train_dataset = self.train_dataset
        val_dataset.save(f"{self.file_name}/val_dataset")
        train_dataset.save(f"{self.file_name}/train_dataset")

        output_file = f"{self.file_name}/event_counts.txt"
        with open(output_file, "w") as f:
            f.write(f"{self.train_events}\n")
            f.write(f"{self.val_events}\n")
        print(f"Dataset Successfully saved")
    
    def load_data(self):
        self.train_dataset = tf.data.Dataset.load(f"{self.file_name}/train_dataset")
        self.val_dataset = tf.data.Dataset.load(f"{self.file_name}/val_dataset")
        with open(f"{self.file_name}/event_counts.txt", "r") as f:
            self.train_events, self.val_events = map(int, f.readlines())


if __name__ == "__main__":
    dataset_2 = Dataset()
    dataset_2.build_dataset()
    dataset_2.save_data()
    dataset_2.load_data()
    print("pocet eventov: ",dataset_2.val_events)
    print("dlzka ", len(dataset_2.val_dataset))
    


