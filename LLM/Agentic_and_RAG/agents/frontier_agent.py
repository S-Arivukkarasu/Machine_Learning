import re 
import os
import litellm
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    """
    An agent that estimates product prices using retrieval-augmented generation (RAG).

    FrontierAgent retrieves products that are semantically similar to a given
    product description from a Chroma vector database. The retrieved products
    and their prices are then provided as context to a language model, which
    generates a price estimate for the target product.

    The agent uses a SentenceTransformer model to generate embeddings for
    similarity search and LiteLLM to communicate with the language model
    hosted behind a Modal endpoint.

    Attributes:
        name (str): Human-readable name of the agent.
        color: Display color inherited from the base Agent class.
        MODEL (str): LiteLLM identifier for the language model used for price estimation.
        model (str): The model identifier used by this agent instance.
        base_url (str): URL of the Modal model endpoint.
        token_id (str): Authentication token ID for the Modal proxy.
        token_secret (str): Authentication token secret for the Modal proxy.
        collection: Chroma collection containing product descriptions and associated price metadata.
        encoder_model (SentenceTransformer): Embedding model used to convert product descriptions into vectors for similarity search.
    """
    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection):
        """
        Initialize the FrontierAgent.

        Loads the Modal endpoint configuration from environment variables,
        stores the Chroma collection, and initializes the SentenceTransformer
        model used for generating embeddings.

        Args:
            collection: A Chroma collection containing product documents and
                price metadata.
        """
        self.log("Initializing Frontier Agent")
        self.model = "openai/Qwen/Qwen3.5-35B-A3B-FP8"
        self.base_url = os.environ['MODAL_ENDPOINT_URL']
        self.token_id = os.environ['MODAL_PROXY_TOKEN_ID']
        self.token_secret = os.environ['MODAL_PROXY_TOKEN_SECRET']
        self.log("Frontier Agent is setting up with Modal")
        self.collection = collection
        self.encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context describing similar products and their prices.

        The similar product descriptions and their corresponding prices are
        formatted into a string that can be included in the language model
        prompt.

        Args:
            similars (List[str]): Descriptions of products similar to the
                product being estimated.
            prices (List[float]): Prices corresponding to the products in
                ``similars``.

        Returns:
            str: Formatted context containing the similar products and their
            prices.
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def message_for(
        self,
        description: str,
        similars: List[str],
        prices: List[float],
    ) -> List[Dict[str, str]]:
        """
        Construct the prompt messages for the language model.

        The generated user message asks the model to estimate the price of
        the target product and includes information about similar products
        retrieved from the Chroma datastore.

        Args:
            description (str): Description of the product whose price should
                be estimated.
            similars (List[str]): Descriptions of similar products retrieved
                from the datastore.
            prices (List[float]): Prices corresponding to the similar
                products.

        Returns:
            List[Dict[str, str]]: A list of chat messages formatted for use
            with the LiteLLM completion API.
        """
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]
    
    def find_similars(self, description: str):
        """
        Find products similar to the given product description.

        The product description is converted into an embedding using the
        SentenceTransformer encoder. The resulting vector is then used to
        query the Chroma collection for the five most similar products.

        Args:
            description (str): Description of the product for which similar
                products should be retrieved.

        Returns:
            tuple:
                A tuple containing two lists:
                - documents (List[str]): Descriptions of the five most
                  similar products.
                - prices (List[float]): Prices associated with those products.
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.encoder_model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        Extract a numeric price from a model response.

        Removes dollar signs and comma separators from the input string,
        then searches for the first integer or floating-point number.

        Args:
            s (str): Model response containing a price.

        Returns:
            float: The first numeric value found in the string. Returns
            ``0.0`` if no number is found.
        """
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Estimate the price of a product using RAG and a language model.

        First retrieves five products similar to the supplied description.
        Their descriptions and prices are then added to the prompt and sent
        to the configured language model through LiteLLM. The numeric price
        is extracted from the model's response and returned.

        Args:
            description (str): Description of the product whose price should
                be estimated.

        Returns:
            float: Estimated price of the product.
        """
        documents, prices = self.find_similars(description=description)
        self.log(
            f"Frontier Agent is about to call {self.model} with context including 5 similar products"
        )
        litellm.api_base = (self.base_url)
        litellm.api_key = (
            f"{self.token_id}."
            f"{self.token_secret}"
            )
        response = litellm.completion(
            model=self.model,
            messages=self.message_for(description, documents, prices),
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            stream=False,
            extra_body={"reasoning_effort": "none"},
            seed=42,
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
