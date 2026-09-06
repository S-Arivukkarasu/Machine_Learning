from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    """
    An agent that uses a trained deep neural network to estimate
    the price of a product from its textual description.

    The agent initializes the neural network inference engine,
    loads the trained model weights, and exposes a simple
    interface for making price predictions.
    """
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize the neural network price prediction agent.

        Creates and configures a ``DeepNeuralNetworkInference``
        instance, then loads the previously trained model weights
        from disk.

        Raises:
            FileNotFoundError: If the model weights file cannot be found.
            RuntimeError: If the neural network or model fails to initialize.
        """
        self.log("Neural Network Agent is initializing")
        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load("/home/alexender/Desktop/Projects/My_projects/Model/deep_neural_network.pth")
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str) -> float:
        """
        Estimate the price of a product from its description.

        Passes the provided product description to the trained
        deep neural network and returns the model's predicted price.

        Args:
            description: A textual description of the product whose
                price should be estimated.

        Returns:
            The estimated product price as a floating-point value.

        Raises:
            ValueError: If the description is invalid or cannot be
                processed by the inference model.
            RuntimeError: If the neural network inference fails.
        """
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
