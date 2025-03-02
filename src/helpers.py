import tensorflow as tf
import numpy as np

@tf.function
def pick_only_data(data, label):
    return data

def extract_data(dataset):
    # Extract all elements from the tf.data.Dataset
    return [x.numpy() for x in dataset]

def make_filter_slice(lower, upper):
    @tf.function
    def _filter_slice(data, target):
        return tf.logical_and(target >= lower, target < upper)
    return _filter_slice


if __name__ == "__main___":
    # Generate bin edges from 70 to 130 (inclusive) for 6 bins: [70,80), [80,90), ..., [120,130)
    bins = np.linspace(70.0, 130.0, num=9)
    filter_functions = [make_filter_slice(lb, ub) for lb, ub in zip(bins[:-1], bins[1:])]
    
    #print(len(filter_functions))
    # Apply each filter function to the training dataset
    #slices = [dataset.train_dataset.filter(fn) for fn in filter_functions]