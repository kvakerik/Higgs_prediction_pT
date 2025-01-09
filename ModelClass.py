import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DatasetClass import Dataset

class Model:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.variables = dataset.variables_higgs
        self.targets = dataset.target_variable
        self.n_train = None
        self.normalizer = None
        self.history = None
        self.model = None
        """
        Model hyperparameters.
        """
        self.input_shape = kwargs.get('input_shape', 4)
        self.model_type = kwargs.get('model_type', "mlp")
        self.activation_function = kwargs.get('activation_function', "sigmoid")
        self.batch_size = kwargs.get('batch_size', 64)
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 100)
        self.n_layers = kwargs.get('n_layers', 4)
        self.initial_learning_rate = kwargs.get('initial_learning_rate', 0.001)
        self.n_epochs = kwargs.get('n_epochs', 10)

    def prepare_dataset(self):
        """
        Prepare the dataset for training and validation. 
        """
        self.dataset.load_data()
        print("dlzka val_datasetu",len(self.dataset.val_dataset))
        self.train_batch = self.dataset.train_dataset.batch(self.batch_size)
        self.val_batch = self.dataset.val_dataset.batch(self.batch_size)

    @staticmethod
    @tf.function
    def pick_only_data(data, label):
        return data

    def create_normalizer(self):
        """
        Create and adapt the normalization layer.
        Args:
            train_dataset: Training dataset.
        """
        self.normalizer = tf.keras.layers.Normalization()
        self.normalizer.adapt(self.train_dataset.map(self.pick_only_data))

    def build_model(self):
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
            initial_learning_rate = self.initial_learning_rate,
            decay_steps = self.n_epochs * self.n_train // self.batch_size,
            alpha = self.initial_learning_rate
        )

        # Compile the model
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.losses.mean_squared_error(),
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
        self.history = history

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

    def plot_output_distributions(self, val_dataset, train_dataset):
        """
        Plot model output distributions for validation and training datasets.
        Args:
            val_dataset: Validation dataset.
            train_dataset: Training dataset.
        """
        y_val = self.model.predict(val_dataset.map(self.pick_only_data))
        y_train = self.model.predict(train_dataset.map(self.pick_only_data))

        plt.figure("Model Output Distribution")
        plt.hist(y_val, bins=100, range=(0, 1), histtype='step', label='Validation Output', density=True)
        plt.hist(y_train, bins=100, range=(0, 1), histtype='step', label='Training Output', density=True)
        plt.xlabel("Model Output")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Output Distribution")
        plt.show()

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_data()
    print((dataset.train_dataset))

    model = Model(input_shape=35, dataset=dataset)
    model.prepare_dataset()


    
    
