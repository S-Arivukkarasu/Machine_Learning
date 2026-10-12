import os
import sys
import json
import logging
import chromadb

from typing import List
from sklearn.manifold import TSNE
from agents.deals import Opportunity
from agents.planning_agent import PlanningAgent

import numpy as np 


# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    """
    Main application framework for running the autonomous deal-finding agents.

    The framework manages the persistent ChromaDB product collection, maintains
    a local memory of previously identified opportunities, initializes the
    planning agent when needed, and executes the end-to-end deal discovery
    workflow.

    It also provides utilities for resetting persisted memory and generating
    dimensionality-reduced product embedding data for visualization.

    Attributes:
        DB: Filesystem path to the persistent ChromaDB vector database.
        MEMORY_FILENAME: Filesystem path to the JSON file used to persist
            discovered opportunities.
        memory: Previously discovered opportunities loaded from persistent
            storage.
        collection: ChromaDB collection containing product embeddings and
            metadata.
        planner: PlanningAgent instance used to execute the deal-finding
            workflow. Initialized lazily when needed.
    """
    DB = "/home/alexender/Desktop/Projects/My_projects/Data/products_vectorstore"
    MEMORY_FILENAME = "/home/alexender/Desktop/Projects/My_projects/Machine_Learning/LLM/Agentic_and_RAG/memory.json"

    def __init__(self):
        """
        Initialize the deal agent framework.

        Configures application logging, creates a persistent ChromaDB client,
        loads previously discovered opportunities from memory, and obtains
        the product collection. The planning agent is initialized lazily by
        ``init_agents_as_needed()``.

        Raises:
            OSError: If the configured database or memory paths cannot be
                accessed.
            json.JSONDecodeError: If the memory file exists but contains
                invalid JSON.
        """
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self.planner = None

    def init_agents_as_needed(self):
        """
        Lazily initialize the planning agent.

        Creates a ``PlanningAgent`` using the configured ChromaDB product
        collection if one has not already been initialized. Subsequent calls
        do nothing when the planner already exists.
        """
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        """
        Load previously discovered opportunities from persistent storage.

        Reads the configured JSON memory file and converts each stored
        opportunity into an ``Opportunity`` instance. If the memory file does
        not exist, an empty list is returned.

        Returns:
            A list of previously discovered ``Opportunity`` objects.

        Raises:
            json.JSONDecodeError: If the memory file contains invalid JSON.
            TypeError: If the stored opportunity data does not match the
                expected ``Opportunity`` structure.
        """
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            # print(opportunities)
            return opportunities
        return []

    def write_memory(self) -> None:
        """
        Persist the current opportunity memory to disk.

        Serializes all opportunities in ``self.memory`` using their Pydantic
        model representation and writes them to the configured JSON memory
        file using indentation for readability.

        Returns:
            None.
        """
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        """
        Reset persisted opportunity memory while retaining the first two entries.

        If the memory file exists, it is loaded and truncated to its first two
        entries. The resulting data is then written back to the memory file.

        This is useful for resetting most of the accumulated agent memory
        while preserving a small number of existing opportunities.

        Returns:
            None.
        """
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(truncated, file, indent=2)

    def log(self, message: str):
        """
        Write a formatted framework message to the application logger.

        The message is displayed using the framework's configured background
        and foreground terminal colors.

        Args:
            message: Message describing the current framework operation.

        Returns:
            None.
        """
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """
        Execute one complete deal-finding workflow.

        Initializes the planning agent if necessary, invokes it with the
        framework's existing opportunity memory, and persists any newly
        discovered opportunity.

        Returns:
            The complete list of opportunities currently stored in memory,
            including any newly discovered opportunity from this run.
        """
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        print(self.memory)
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        """
        Retrieve and reduce product embeddings for visualization.

        Loads product embeddings, documents, and category metadata from the
        ChromaDB collection. The embeddings are reduced to three dimensions
        using t-SNE, and each product is assigned a color based on its
        category.

        Args:
            max_datapoints: Maximum number of product records to retrieve
                from ChromaDB. Defaults to 2000.

        Returns:
            A tuple containing:
                - documents: Product descriptions associated with the
                  retrieved embeddings.
                - reduced_vectors: NumPy array containing the three-dimensional
                  t-SNE representation of the product embeddings.
                - colors: List of plot colors corresponding to each product's
                  category.

        Raises:
            ValueError: If a product's category is not present in
                ``CATEGORIES`` or if the retrieved embeddings cannot be
                processed by t-SNE.
        """
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()