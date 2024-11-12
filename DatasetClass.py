import os
import tensorflow as tf
import uproot
import glob
import vector
import awkward as ak
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

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
]

class DatasetConstructor():
    def __init__(self): 
        self.variable_names = variables_higgs 
        self.path_sig = path_sig
        self.path_bkg = path_bkg

    def importFiles(self):
        files_sig = glob.glob(self.path_sig)
        files_bkg = glob.glob(self.path_bkg)
        #print(f"imported signal files: {files_sig}")
        #print("--"*50)
        #print(f"imported background files: {files_bkg}")
        return files_sig, files_bkg

    def loadAndConvertToTensors(self):
        # Import files
        files_sig, files_bkg = self.importFiles()
        all_files = files_sig + files_bkg  # Combine signal and background

        tensors = []
        n_events = []

        # Process each file individually
        for file in all_files:
            print("Reading file", file)
            f = uproot.open(file)['NOMINAL']
            data = f.arrays(self.variable_names, library="ak")

            tensors_var = []
            n_evt = 0

            # Process each variable and convert directly to tensor
            for var in self.variable_names:
                if 'p4' in var:
                    # Extract 4-vector components
                    p4 = vector.zip({
                        'x': data[var]['fP']['fX'], 
                        'y': data[var]['fP']['fY'], 
                        'z': data[var]['fP']['fZ'],
                        't': data[var]['fE']
                    })
                    tensors_var.extend([tf.constant(ak.to_numpy(p4.rho), dtype=tf.float32),  # pt
                                        tf.constant(ak.to_numpy(p4.eta), dtype=tf.float32),  # eta
                                        tf.constant(ak.to_numpy(p4.phi), dtype=tf.float32),  # phi
                                        tf.constant(ak.to_numpy(p4.tau), dtype=tf.float32)]) # mass
                else:
                    tensors_var.append(tf.constant(ak.to_numpy(data[var]), dtype=tf.float32))

            # Stack tensors for each file
            tensor_stack = tf.stack(tensors_var, axis=1)
            tensors.append(tensor_stack)
            n_evt += tensor_stack.shape[0]

            print(f"File {file} processed with tensor shape: {tensor_stack.shape}")
            n_events.append(n_evt)

        #print("Total tensors processed:", tensors)
        print("Total events", np.sum(n_events))
        
        return tensors, n_events

if __name__ == "__main__":
    datasetConstructor = DatasetConstructor()
    tensors, n_events = datasetConstructor.loadAndConvertToTensors()
