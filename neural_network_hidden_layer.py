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
        self.activations = np.zeros(shape=(num_units,))
        self.activation_output = float
        self.sum_input_activation = np.zeros(shape=(num_units,))
        self.delta_i = []

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.
        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 380

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        # input dim should be number of attributes in x_train
        # implementere lag klasse: inneholder alle noder i et av lagene
        # ta fra input, returnere output verdi
        self.input_dim = input_dim
        # + 1 pga bias
        self.num_units = 25
        # Number of hidden units if hidden_layer = True.
        # We want 31 weights per 25 hidden units
        self.input_weights = np.random.uniform(low=-1, high=1, size=(self.num_units,self.input_dim+1))
        self.output_weights = np.random.uniform(low=-1, high=1, size=(self.num_units,))

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
        
        
        layer = Layer(self.num_units, self.input_dim)

        for i in range(self.epochs):
            for x_j,y_j in zip(examples,y_train):
                # FORWARD PROPAGATION
                # Input weights
                bias = np.array([1])                 # Fikser bias
                activation_input = np.concatenate((x_j, bias))    # a_i <- x_i
                for hidden_node in range(self.num_units):
                    # Calculating w_ij * a_i
                    input = np.multiply(self.input_weights[hidden_node], activation_input)
                    # Calculating the sum of w_ij * a_i
                    sum_input = sum(input)
                    # Calculating the activation for one node in the hidden layer
                    activation_node = sigmoid(sum_input)
                    # Saving stuff in layer
                    layer.activations[hidden_node] = activation_node

                # Output value
                layer.activation_output = sigmoid(sum(np.multiply(layer.activations, self.output_weights)))
                
                # BACKWARD PROPAGATION
                # Bare en output node, driter i for-løkken på linje 12
                sigmoid_prime_output = sigmoid_prime(layer.activation_output)
                # SPØR: Hva er a_j her, linje 13? det blir det riktig å bruke activation_output?
                delta_j = sigmoid_prime_output * (y_j-layer.activation_output)

                # SPØR: Hva gjør delta i på linje 16??? Hvordan påvirker den vektene i nettverket? 
                # for alle noder i ikke output laget -> gjør delta i greiene på linje 16
                # for i in range(self.input_dim+self.num_units):

                # UPDATE WEIGHTS
                # weight_input = numpy.array(25,31)
                # alle 25 noder har 31 input vekter
                # all_weights = np.concatenate((self.input_weights, self.output_weights),axis=0)
                # Calculating the updating factor for each layer
                update_input = self.lr * activation_input * delta_j
                update_output = self.lr * layer.activations * delta_j
                # Updating the weights in the network
                for k in range(len(self.input_weights)):
                    self.input_weights[k] = np.add(self.input_weights[k], update_input)
                self.output_weights = np.add(self.output_weights, update_output)



                
    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # Setting up the bias
        bias = np.array([1])                 # Fikser bias
        activation_input = np.concatenate((x, bias))    # a_i <- x_i
        
        activations = np.zeros(shape=(self.num_units,))

        for hidden_node in range(self.num_units):
            # Calculating w_ij * a_i
            input = np.multiply(self.input_weights[hidden_node], activation_input)
            # Calculating the sum of w_ij * a_i
            sum_input = sum(input)
            # Calculating the activation for one node in the hidden layer
            activation_node = sigmoid(sum_input)
            # Saving stuff in layer
            activations[hidden_node] = activation_node

        # Output value
        activation_output = sigmoid(sum(np.multiply(activations, self.output_weights)))
        return  activation_output
    

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

    # def test_perceptron(self) -> None:
    #     """Run this method to see if Part 1 is implemented correctly."""

    #     self.network = self.nn_class(self.n_features, False)
    #     accuracy = self.get_accuracy()
    #     self.assertTrue(accuracy > self.threshold,
    #                     'This implementation is most likely wrong since '
    #                     f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
