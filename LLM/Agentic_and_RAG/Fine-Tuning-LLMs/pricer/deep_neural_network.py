# Importing Libraries

import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


class ResidualBlock(nn.Module):
    """
    A single residual block used in a deep neural network.

    This block applies:
    - Linear transformation
    - Layer normalization
    - ReLU activation
    - Dropout
    - Another linear transformation + normalization

    A skip (residual) connection is added from input to output
    to help with gradient flow and stabilize deep networks.
    """
    def __init__(self, hidden_size, dropout_prob):
        """
        Initialize the residual block.

        Args:
            hidden_size (int): Number of neurons in the hidden layer.
            dropout_prob (float): Dropout probability for regularization.
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
        Forward pass through the residual block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, hidden_size)

        Returns:
            Tensor: Output tensor after residual addition and activation
        """
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    """
    A deep fully connected neural network with residual connections.

    Architecture:
    - Input layer (Linear + LayerNorm + ReLU + Dropout)
    - Multiple residual blocks
    - Output layer (Linear -> 1 value)

    Designed for regression tasks (predicting a continuous value).
    """
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        """
        Initialize the deep neural network.

        Args:
            input_size (int): Number of input features.
            num_layers (int): Total number of layers including input/output.
            hidden_size (int): Number of neurons per hidden layer.
            dropout_prob (float): Dropout probability.
        """
        super(DeepNeuralNetwork, self).__init__()

        # First layer transforms input into hidden representation
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Final output layer (regression)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the full network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor: Predicted values of shape (batch_size, 1)
        """
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class DeepNeuralNetworkRunner:
    """
    Utility class to handle:
    - Data preprocessing
    - Model setup
    - Training loop
    - Validation
    - Saving/loading
    - Inference

    This class wraps the full pipeline for training a regression model
    on text data (summaries → price prediction).
    """
    
    def __init__(self, train, val):
        """
        Initialize runner with training and validation datasets.

        Args:
            train (list): Training dataset (objects with .summary and .price)
            val (list): Validation dataset (same structure as train)
        """
        self.train_data = train
        self.val_data = val
        self.vectorizer = None
        self.model = None
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.train_loader = None
        self.y_mean = None
        self.y_std = None

        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        """
        Prepare data, initialize model, and configure training components.

        Steps:
        - Vectorize text using HashingVectorizer
        - Convert data into tensors
        - Apply log transform and normalization on targets
        - Initialize model, optimizer, scheduler
        - Create DataLoader for batching
        """
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        # Prepare training data
        train_documents = [item.summary for item in self.train_data]
        X_train_np = self.vectorizer.fit_transform(train_documents)
        self.X_train = torch.FloatTensor(X_train_np.toarray())
        y_train_np = np.array([float(item.price) for item in self.train_data])
        self.y_train = torch.FloatTensor(y_train_np).unsqueeze(1)

        # Prepare validation data
        val_documents = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_documents)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        y_val_np = np.array([float(item.price) for item in self.val_data])
        self.y_val = torch.FloatTensor(y_val_np).unsqueeze(1)

        # Log transform targets to stabilize training
        y_train_log = torch.log(self.y_train + 1)
        y_val_log = torch.log(self.y_val + 1)

        # Normalize targets
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        self.y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        # Initialize model
        self.model = DeepNeuralNetwork(self.X_train.shape[1])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {total_params:,} parameters")

        # Select device (GPU / MPS / CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        self.model.to(self.device)

        # Loss and optimizer
        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        # DataLoader for batching
        self.train_dataset = TensorDataset(self.X_train, self.y_train_norm)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def train(self, epochs=5):
        """
        Train the model.

        Args:
            epochs (int): Number of training epochs.

        Performs:
        - Forward pass
        - Loss computation
        - Backpropagation
        - Gradient clipping
        - Validation after each epoch
        - Learning rate scheduling
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in tqdm(self.train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val.to(self.device))
                val_loss = self.loss_function(val_outputs, self.y_val_norm.to(self.device))

                # Convert back to original scale for meaningful metrics
                val_outputs_orig = torch.exp(val_outputs * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_outputs_orig - self.y_val.to(self.device)).mean()

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Val mean absolute error: ${mae.item():.2f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

    def save(self, path):
        """
        Save the trained model weights.

        Args:
            path (str): File path to save model state.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path, device="mps"):
        """
        Load model weights from disk.

        Args:
            path (str): Path to saved model.
            device (str): Device to map model onto.
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(self.device)

    def inference(self, item):
        """
        Run inference on a single data item.

        Args:
            item: Object with a `.summary` attribute.

        Returns:
            float: Predicted price (non-negative).
        """
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]

            # Convert back to original scale
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)