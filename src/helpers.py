import tensorflow as tf

@tf.function
def pick_only_data(data, label):
    return data

def extract_data(dataset):
    # Extract all elements from the tf.data.Dataset
    return [x.numpy() for x in dataset]