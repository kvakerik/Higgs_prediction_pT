import os
import tensorflow as tf
import uproot
import glob
import vector
import awkward as ak

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
    "ditau_deta","ditau_dphi","ditau_dr","ditau_higgspt","ditau_scal_sum_pt", #"ditau_mmc_mlm_m",
    "jet_0_p4",
    "jet_1_p4",
    "dijet_p4", # fixme add dEta
    "met_p4", 
    "n_jets","n_jets_30","n_jets_40","n_electrons","n_muons","n_taus",
]

class DatasetConstructor():
    def __init__(self): 
        self.variable_names = variables_higgs 
        self.path_sig = path_sig
        self.path_bkg = path_bkg
        self.arrays = []
        self.tensors = []
        self.n_events = []

    def importFiles(self):
        files_sig = glob.glob(self.path_sig)
        files_bkg = glob.glob(self.path_bkg)
        print(f"imported {files_sig}")
        print("--"*50)
        print(f"imported {files_bkg}")
              
        return files_sig, files_bkg
    
    def processTTrees(self):
        file_paths = self.importFiles()
        for files in file_paths:
            self.arrays.append([])
            for file in files:
                print("Reading file", file)
                f = uproot.open(file)['NOMINAL']
                data = f.arrays(self.variable_names, library="ak")
                arr = []
                for var in self.variable_names:
                    if 'p4' in var:
                        # We need to extract the 4-vector pt, eta, phi, mass
                        p4 = vector.zip({'x':data[var]['fP']['fX'], 
                                        'y':data[var]['fP']['fY'], 
                                        'z':data[var]['fP']['fZ'],
                                        't':data[var]['fE']})
                        
                        arr.append(p4.rho) # pt
                        arr.append(p4.eta) # eta
                        arr.append(p4.phi) # phi
                        arr.append(p4.tau) # mass
                    
                    else:
                        arr.append(data[var])

                self.arrays[-1].append(arr)    

        print(len(self.arrays[0]))
        print(len(self.arrays[1]))





if __name__ == "__main__":
    datasetConstructor = DatasetConstructor()
    datasetConstructor.processTTrees()


