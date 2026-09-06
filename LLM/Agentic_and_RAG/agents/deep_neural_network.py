# Importing Libraries

import numpy as np
from tqdm.notebook import tqdm
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


class ResidualBlock(nn.Module):
    """
    Implements a residual neural network block.

    The block applies two fully connected layers with layer normalization,
    ReLU activation, and dropout. The original input is added to the
    transformed output through a residual (skip) connection, followed by
    a final ReLU activation.

    Parameters
    ----------
    hidden_size : int
        Number of input and output features for the residual block.
    dropout_prob : float
        Probability used by the dropout layer to randomly zero elements
        during training.

    Attributes
    ----------
    block : nn.Sequential
        Sequential collection of linear, normalization, activation, and
        dropout layers.
    relu : nn.ReLU
        ReLU activation function applied after the residual connection.
    """
    def __init__(self, hidden_size, dropout_prob):
        """
        Initialize the residual block.

        Parameters
        ----------
        hidden_size : int
            Number of features in the input and output tensors.
        dropout_prob : float
            Dropout probability used during training.
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Perform a forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the hidden representations.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual transformation
            and ReLU activation.
        """
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)
    

class DeepNeuralNetwork(nn.Module):
    """
    Implements a deep fully connected neural network with residual blocks.

    The network consists of an input layer, multiple residual blocks,
    and a final linear output layer. Layer normalization, ReLU activation,
    dropout, and residual connections are used to improve training stability
    and model performance.

    Parameters
    ----------
    input_size : int
        Number of features in the input vector.
    num_layers : int, default=10
        Total number of layers used in the network.
    hidden_size : int, default=4096
        Number of neurons in each hidden layer.
    dropout_prob : float, default=0.2
        Probability used by dropout layers during training.

    Attributes
    ----------
    input_layer : nn.Sequential
        Initial transformation applied to the input features.
    residual_blocks : nn.ModuleList
        Collection of residual blocks used for deep feature learning.
    output_layer : nn.Linear
        Final linear layer that produces a single prediction.
    """
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        """
        Initialize the deep neural network.

        Parameters
        ----------
        input_size : int
            Number of features in the input data.
        num_layers : int, default=10
            Number of layers in the network.
        hidden_size : int, default=4096
            Number of hidden units in each layer.
        dropout_prob : float, default=0.2
            Dropout probability used in the network.
        """
        super(DeepNeuralNetwork, self).__init__()

        # First Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size=hidden_size, dropout_prob=dropout_prob))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the feature representations.

        Returns
        -------
        torch.Tensor
            Tensor containing the model's scalar predictions.
        """
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_layer(x)


Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    """
    Provides an inference pipeline for the trained deep neural network.

    This class is responsible for initializing the text vectorizer,
    creating the neural network, selecting the appropriate computation
    device, loading trained model weights, and generating predictions
    from input text.

    Attributes
    ----------
    vectorizer : HashingVectorizer or None
        Hashing vectorizer used to convert text into numerical features.
    model : DeepNeuralNetwork or None
        Neural network used for generating predictions.
    device : torch.device or None
        Device used for model inference, such as CUDA, MPS, or CPU.
    """
    def __init__(self):
        """
        Initialize the neural network inference pipeline.

        Initializes the vectorizer, model, and device attributes.
        Random seeds are also initialized to improve reproducibility
        of NumPy and PyTorch operations.
        """
        self.vectorizer = None
        self.model = None
        self.device = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        """
        Set up the text vectorizer, neural network, and computation device.

        Creates a HashingVectorizer with 5,000 features and English
        stop-word removal. The method also initializes the neural network
        and automatically selects CUDA, Apple MPS, or CPU depending on
        hardware availability.

        Returns
        -------
        None
            This method initializes the inference components in place.
        """
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        self.model.to(self.device)

    def load(self, path):
        """
        Load trained neural network weights from a file.

        The saved PyTorch state dictionary is loaded into the neural
        network using the currently selected computation device.

        Parameters
        ----------
        path : str
            File path containing the saved model state dictionary.

        Returns
        -------
        None
            The model parameters are loaded directly into ``self.model``.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
    
    def inference(self, text):
        """
        Generate a prediction for the given input text.

        The input text is converted into a numerical feature vector using
        the HashingVectorizer and passed through the trained neural network.
        The model output is then denormalized using the predefined target
        mean and standard deviation and transformed back from logarithmic
        space.

        Parameters
        ----------
        text : str
            Input text to be used for generating a prediction.

        Returns
        -------
        float
            Non-negative prediction generated by the neural network.

        Notes
        -----
        The prediction is converted back to its original scale using:

        ``exp(pred * Y_STD + Y_MEAN) - 1``

        The returned value is constrained to be at least zero.
        """
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
            result = result.item()

        return max(0, result)