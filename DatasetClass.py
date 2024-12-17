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
    def __init__(self,batch_size=64,train_fraction=0.8): 
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

    def buildDataset(self, plot_variables : bool = False):
        # Import files
        files_sig, files_bkg = self.importFiles()
        all_files = files_sig + files_bkg  # Combine signal and background

        print("All files ",len(all_files))
        
        arrays = []
        arrays_truth = []
        
        # Premena awkward arrays na tensorflow tensors s truth vektormi
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
                    # We need to extract the 4-vector pt, eta, phi, mass
                    p4 = vector.zip({'x':data[var]['fP']['fX'], 
                                    'y':data[var]['fP']['fY'], 
                                    'z':data[var]['fP']['fZ'],
                                    't':data[var]['fE']})
                    
                    arr.append(p4.rho) # pt
                    arr.append(p4.eta) # eta
                    arr.append(p4.phi) # phi
                    arr.append(p4.tau) # mass

                elif (var == "boson_0_truth_p4"):
                    target_p4 = vector.zip({'x':data[var]['fP']['fX'], 
                                    'y':data[var]['fP']['fY'], 
                                    'z':data[var]['fP']['fZ'],
                                    't':data[var]['fE']})
                    
                    arr_truth.append(target_p4.rho) # pt
                    arr_truth.append(target_p4.eta) # eta
                    arr_truth.append(target_p4.phi) # phi
                    arr_truth.append(target_p4.tau) # mass
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
                    else:
                        print(f"Skipping plot for {var} - {name} due to non-numeric or empty data.")
            else:
                print("Plot variables is turned off")
                            
            for j, (arr_file, arr_truth_file) in enumerate(zip(arrays, arrays_truth)):
                n_evt = 0
                tensors_var = []
                truths_var = []

                # Iterácia cez jednotlivé premenné a truth vektory
                for arr_var in arr_file:
                    # Konverzia premenných na tensorflow tenzory
                    tensor = tf.constant(ak.to_numpy(arr_var), dtype=tf.float32)
                    tensors_var.append(tensor)
                
                for truth_var in arr_truth_file:
                    # Konverzia truth na tensorflow tensor
                    truth_tensor = tf.constant(ak.to_numpy(truth_var), dtype=tf.float32)
                    truths_var.append(truth_tensor)

                # Stack premenných
                tensor_stack = tf.stack(tensors_var, axis=1)  # Premenné (napr. 35 stĺpcov)
                truth_stack = tf.stack(truths_var, axis=1)    # Pravdivostné vektory (napr. 4 stĺpce)

            # Pridanie do zoznamov
            tensors.append(tensor_stack)
            truth_vectors.append(truth_stack)

            n_evt += tensor_stack.shape[0]

            # Výstup pre kontrolu
            print(j,tensor_stack.shape, truth_stack.shape)

            n_events.append(n_evt)
 
        #print("Total tensors processed:", tensors)
        print("Total events", np.sum(n_events))
        #print(f"Length of tensor lists {len(tensors)}")
        
        datasets = []

        for tensor_file, truth_file in zip(tensors,truth_vectors):
                dataset_sample = tf.data.Dataset.from_tensor_slices((tensor_file,truth_file))
                datasets.append(dataset_sample)
        print("Datasets_len",len(datasets))
        print(datasets)
        
        train_datasets=[]
        val_datasets=[]
        
        for dataset, dataset_size in zip(datasets,n_events):
            
            # Determine datasets split sizes
            train_size = int(self.train_fraction * dataset_size)

            # Split datasets to train and validation data
            train_dataset = datasets.take(train_size)
            val_dataset = datasets.skip(train_size)

            train_dataset = train_dataset.batch(self.batch_size)
            val_dataset = val_dataset.batch(self.batch_size)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            
            print(f"Dataset size: {dataset_size}, Training size: {train_size}, Validation size: {dataset_size - train_size}")

        print("train",len(train_datasets))
        print("val",len(val_datasets))
    
        weights_list = []
        for tensor, total_events in zip(tensors, n_events):
            weights = [tensor.shape[0] / total_events]
            weights_list.extend(weights)
        print("weights_list_len",len(weights_list))
        
        dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights_list)
        print("Dataset Successfully imported")
        print(type(dataset))
        for x in dataset.take(1):
            print(x)    
        
        return dataset, n_events
        
if __name__ == "__main__":
    datasetConstructor = DatasetConstructor()
    dataset, n_events = datasetConstructor.buildDataset(plot_variables=False)

    
   

