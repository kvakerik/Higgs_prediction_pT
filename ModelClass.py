import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DatasetClass import DatasetConstructor

class ModelConstructor:
    def __init__(self, input_shape, variables, model_type = "mlp", activation_function = "sigmoid", batch_size=64, hidden_layer_size=100, n_layers=4, initial_learning_rate=0.001, n_epochs=10):
        self.input_shape = input_shape
        self.variables = variables
        self.model_type = model_type
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        self.initial_learning_rate = initial_learning_rate
        self.n_epochs = n_epochs
        self.model = None
        self.normalizer = None

    def prepare_dataset(self, use_saved_data = True):
        """
        Prepare datasets for training and validation.
        Args:
            dataframe: Input pandas DataFrame.
            label_col: Name of the label column.
            train_fraction: Fraction of the dataset used for training.
        """

        #TODO import data from datasetConstructor and work with target and train data 
        # Use DatasetConstructor to build the dataset
        dataset_constructor = DatasetConstructor()
        if use_saved_data:

            val_dataset = tf.data.Dataset.load("data/val_dataset")
            train_dataset =tf.data.Dataset.load("data/train_dataset")
            val_events, train_events = np.loadtxt("data/event_counts.txt")
        else:
            val_dataset, train_dataset, val_events, train_events = dataset_constructor.buildDataset(plot_variables=False, save_dataset=False)

        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)

        return train_dataset, val_dataset, val_events, train_events 

    @staticmethod
    @tf.function
    def pick_only_data(data, label):
        return data

    def create_normalizer(self, train_dataset):
        """
        Create and adapt the normalization layer.
        Args:
            train_dataset: Training dataset.
        """
        self.normalizer = tf.keras.layers.Normalization()
        self.normalizer.adapt(train_dataset.map(self.pick_only_data))

    def build_model(self, n_train):
        """
        Build a deep neural network model with learning rate decay.
        Args:
            n_train: Number of training samples (used for learning rate decay).
        """
        input_layer = tf.keras.layers.Input(shape=(len(self.variables),))
        layer = self.normalizer(input_layer)

        for i in range(self.n_layers):
            layer = tf.keras.layers.Dense(
                self.hidden_layer_size // self.n_layers,
                activation="relu"
            )(layer)

        output_layer = tf.keras.layers.Dense(1, activation=self.activation_function)(layer)

        # Define learning rate decay schedule
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.n_epochs * n_train // self.batch_size,
            alpha=self.initial_learning_rate
        )

        # Compile the model
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.BinaryAccuracy(), tf.metrics.Recall(), tf.metrics.Precision()]
        )

    def train_model(self, train_dataset, val_dataset, epochs=None):
        """
        Train the model.
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            epochs: Number of training epochs (overrides default if provided).
        """
        if not self.model:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        epochs = epochs or self.n_epochs
        history = self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        return history

    def evaluate_model(self, dataset):
        """
        Evaluate the model.
        Args:
            dataset: Dataset to evaluate on.
        """
        return self.model.evaluate(dataset)

    def plot_history(self, history):
        """
        Plot training history.
        Args:
            history: Training history object from model.fit().
        """
        plt.figure("Training Loss")
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Training and Validation Loss")

        plt.figure("Training Accuracy")
        plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Training and Validation Accuracy")
        plt.show()

    def plot_output_distributions(self, val_dataset, train_dataset):
        """
        Plot model output distributions for validation and training datasets.
        Args:
            val_dataset: Validation dataset.
            train_dataset: Training dataset.
        """
        y_val = self.model.predict(val_dataset.map(self.pick_only_data).unbatch().batch(1024))
        y_train = self.model.predict(train_dataset.map(self.pick_only_data).unbatch().batch(1024))

        plt.figure("Model Output Distribution")
        plt.hist(y_val, bins=100, range=(0, 1), histtype='step', label='Validation Output', density=True)
        plt.hist(y_train, bins=100, range=(0, 1), histtype='step', label='Training Output', density=True)
        plt.xlabel("Model Output")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Output Distribution")
        plt.show()

if __name__ == "__main__":
    modelConstructor = ModelConstructor()
    train_dataset, val_dataset = modelConstructor.prepare_dataset()
