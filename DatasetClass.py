import os
import tensorflow as tf
import uproot
import glob
import vector
import awkward as ak
# import matplotlib.pyplot as plt 
import numpy as np
from src.helpers import pick_only_data, extract_data, make_filter_slice

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

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
                    "n_jets","n_jets_30","n_jets_40","n_electrons","n_muons","n_taus","ditau_mmc_mlm_m"
                    ]
        
        default_target_variable = "truth_boson_p4"
        
        self.variables_higgs = kwargs.get('variables_higgs', default_variables_higgs)
        self.target_variable = kwargs.get('target_variable', default_target_variable)
        if self.target_variable not in self.variables_higgs:
            self.variables_higgs.append(self.target_variable)

        self.train_fraction = kwargs.get('train_fraction', 0.8)
        self.file_paths = kwargs.get('file_paths', "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*H125*.root")
        self.val_dataset = None
        self.train_dataset = None
        self.val_events = None
        self.train_events = None
        self.slices = None

    def __call__(self):
        self.load_data()

    def importFiles(self): 
        print("Importing Root files",self.file_paths)
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
        for file in all_files:
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
        dev_datasets = []
        
        train_size_dataset = []
        val_size_dataset = []
        dev_size_dataset = []

        for dataset, dataset_size in zip(datasets, n_events):
            # Determine datasets split sizes
            train_size = int(round(self.train_fraction * dataset_size))
            remaining_size = dataset_size - train_size
            val_size = round(remaining_size / 2)
            dev_size = dataset_size - train_size - val_size

            if val_size > 0 and train_size > 0 and dev_size > 0:
                # Split datasets into train and validation data
                train_dataset = dataset.take(train_size)
                val_dataset = dataset.skip(train_size).take(val_size)
                dev_dataset = dataset.skip(train_size + val_size).take(dev_size)

                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)
                dev_datasets.append(dev_dataset)

                train_size_dataset.append(train_size)
                val_size_dataset.append(val_size)
                dev_size_dataset.append(dev_size)
            else:
                print(f"Skipping file with dataset size: {dataset_size}")

        #print(f"Dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}")
        #print(len(train_datasets), len(val_datasets))

        train_events = sum(train_size_dataset)
        val_events = sum(val_size_dataset)
        dev_events = sum(dev_size_dataset)
        
        weights_list_train = [size / train_events for size in train_size_dataset]
        weights_list_val = [size / val_events for size in val_size_dataset]
        weights_list_dev = [size / dev_events for size in dev_size_dataset]

        print("Dataset Successfully weighted")
        
        if len(val_datasets) > 0:
            self.val_dataset = tf.data.Dataset.sample_from_datasets(val_datasets, weights=weights_list_val)

        if len(train_datasets) > 0:
            self.train_dataset = tf.data.Dataset.sample_from_datasets(train_datasets, weights=weights_list_train)

        if len(dev_datasets) > 0:
            self.dev_dataset = tf.data.Dataset.sample_from_datasets(dev_datasets, weights=weights_list_dev)

        self.val_events = val_events
        self.train_events = train_events
        self.dev_events = dev_events
    
    def save_data(self, file_name= "data_mmc"):  
        print("saving dataset")
        os.makedirs(f"{file_name}", exist_ok=True)  # Ensure 'data' directory exists
        val_dataset = self.val_dataset
        train_dataset = self.train_dataset
        dev_dataset = self.dev_dataset

        val_dataset.save(f"{file_name}/val_dataset")
        train_dataset.save(f"{file_name}/train_dataset")
        dev_dataset.save(f"{file_name}/dev_dataset")

        output_file = f"{file_name}/event_counts.txt"
        with open(output_file, "w") as f:
            f.write(f"{self.train_events}\n")
            f.write(f"{self.val_events}\n")
            f.write(f"{self.dev_events}\n")
        print(f"Dataset Successfully saved")

    def load_data(self, file_name = "data_mmc"):
        self.train_dataset = tf.data.Dataset.load(f"{file_name}/train_dataset")
        self.val_dataset = tf.data.Dataset.load(f"{file_name}/val_dataset")
        self.dev_dataset = tf.data.Dataset.load(f"{file_name}/dev_dataset")
        with open(f"{file_name}/event_counts.txt", "r") as f:
            self.train_events, self.val_events, self.dev_events = map(int, f.readlines())

    def plot_distribution(self):
        # Extract data from the TensorFlow datasets
        y_val = extract_data(self.val_dataset.map(pick_only_data))
        y_train = extract_data(self.train_dataset.map(pick_only_data))

        plt.figure(figsize=(10, 6))
        plt.hist(y_val, bins=100, range=(0, 1), histtype='step', label='Validation Output', density=True)
        plt.hist(y_train, bins=100, range=(0, 1), histtype='step', label='Training Output', density=True)
        plt.xlabel("Input Data")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Input Distribution")
        plt.show()

class DatasetPt(Dataset):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def get_phi_mask(self):
        mask = []
        for var in self.variables_higgs: 
            #print(var)
            if ('p4' in var) and (var != self.target_variable):
                mask.append(False)  # pt
                mask.append(False) # eta
                mask.append(True) # phi
                mask.append(False) # mass

            elif (var == self.target_variable):
                pass
            else:
                if 'phi' in var:
                    mask.append(True)
                else:
                    mask.append(False)
        return mask

    def load_data(self):
        super().load_data(file_name = "data_mmc")

        ## add augmentation
        
        ## pick pt
        @tf.function
        def pick_pt(data, targets):
            return data, targets[0]
        
        self.train_dataset = self.train_dataset.map(pick_pt)
        self.val_dataset = self.val_dataset.map(pick_pt)
        self.dev_dataset = self.dev_dataset.map(pick_pt)

if __name__ == "__main__":
    dataset = DatasetMass()
    dataset.load_data(file_name = "data")
    dataset.augment_data_phi()

    print(dataset.train_events)
    print(dataset.val_events)
    print(dataset.dev_events)



    

   
    


