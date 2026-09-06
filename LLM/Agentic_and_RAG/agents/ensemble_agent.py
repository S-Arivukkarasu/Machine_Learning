import re
from agents.agent import Agent
from agents.preprocessor import Preprocessor
from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent
from agents.neural_network_agent import NeuralNetworkAgent


class EnsembleAgent(Agent):
    """
    An ensemble pricing agent that combines predictions from multiple models.

    The ensemble uses a preprocessor to normalize the product description and
    then obtains independent price estimates from a specialist model, a
    frontier model, and a neural network model. These predictions are combined
    using fixed weights to produce the final estimated price.

    Attributes:
        name (str): Display name of the agent.
        color: Color used when logging or displaying the agent.
        specialist (SpecialistAgent): Model specialized in product pricing.
        frontier (FrontierAgent): Frontier model used to generate a price
            estimate based on the supplied collection.
        neural_network (NeuralNetworkAgent): Neural network pricing model.
        preprocessor (Preprocessor): Component responsible for preprocessing
            product descriptions before they are passed to the models.
    """
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Initialize the ensemble agent and its component models.

        Args:
            collection: Collection of products or data used by the frontier
                agent when generating price estimates.
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection=collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Estimate the price of a product using the ensemble of models.

        The product description is first preprocessed and stripped of
        Markdown-style headings. The cleaned description is then passed to
        the specialist, frontier, and neural network models. Their predictions
        are combined using fixed weights, with the frontier model contributing
        80% and the other two models contributing 10% each.

        Args:
            description (str): Textual description of the product to price.

        Returns:
            float: The ensemble's estimated price for the product.
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        rewrite = re.sub(r"(?m)^#.*\n?", "", rewrite)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        combined = frontier*0.8 + specialist*0.1 + neural_network*0.1
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
