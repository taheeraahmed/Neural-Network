# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Layer:
    def __init__(self, num_units: int, input_dim: int):
        #self.num_units = num_units
        # Skal lagre input til alle noder i en liste? Burde være input_dim lang
        self.activations = []
        self.output = float
        self.output_weights = np.random.uniform(low=-1, high=1, size=(num_units,))
        self.input_weights = np.random.uniform(low=-1, high=1, size=(num_units,(input_dim+1)))

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        self.lr = 1e-3
        self.epochs = 380

        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim

        self.weights = np.random.uniform(low=-1, high=1, size=(input_dim+1,))
        self.hidden_units = 0
        self.num_layers = 0

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data/data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        
        # Initializing everything
        examples = self.x_train
        y_train = self.y_train
        
        for i in range(self.epochs):
            for x_j,y_j in zip(examples,y_train):
                # FORWARD PROPAGATION
                # Input layer
                weights = self.weights
                bias = np.array([1])                 # Fikser bias
                x_j = np.concatenate((x_j, bias))    # a_i <- x_i
                activation_input = x_j * weights     # a_i <- x_i

                in_j = sum(activation_input)
                a_j= sigmoid(in_j)
                # Ettersom dette er perceptron dropper å iterere gjennom lag fordi lol 
                # Dropper linje 7 - 11 enn så lenge

                # BACKWARD PROPAGATION
                g_prime = sigmoid_prime(in_j)
                temp = y_j-a_j
                delta_j = g_prime * temp

                # UPDATE WEIGHTS
                i = 0
                for w_i,a_i in zip(weights,x_j):
                    w_i = w_i + (self.lr * a_i * delta_j)
                    weights[i] = w_i
                    i += 1
                self.weights = weights

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        arr = np.array([1])
        x = np.concatenate((x, arr))
        return sigmoid(sum(x*self.weights))

class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train() 
        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    # def test_one_hidden(self) -> None:
    #     """Run this method to see if Part 2 is implemented correctly."""

    #     self.network = self.nn_class(self.n_features, True)
    #     accuracy = self.get_accuracy()
    #     self.assertTrue(accuracy > self.threshold,
    #                     'This implementation is most likely wrong since '
    #                     f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
