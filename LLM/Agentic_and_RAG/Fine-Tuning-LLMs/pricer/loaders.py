from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from pricer.parser import parse
import os

CHUNK_SIZE = 1000

cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


class ItemLoader:
    """
    ItemLoader is responsible for loading, parsing, and processing
    Amazon dataset metadata in parallel.

    It:
    - Loads a dataset from Hugging Face
    - Breaks it into chunks
    - Processes each datapoint using a parser
    - Uses multiprocessing to speed up execution

    Attributes:
        category (str): Dataset category (e.g., 'Electronics', 'Books')
        dataset (Dataset): Loaded Hugging Face dataset
    """
    def __init__(self, category):
        """
        Initialize the loader with a specific dataset category.

        Args:
            category (str): Name of the dataset category.
        """
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        
        Convert a single datapoint into an Item.

        This method uses the external `parse` function to transform
        raw dataset entries into structured items.

        Args:
            datapoint (dict): A single dataset record.

        Returns:
            Item or None:
                - Returns parsed Item if successful
                - Returns None if datapoint should be skipped
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        
        Process a chunk of datapoints.

        Applies `from_datapoint` to each datapoint in the chunk
        and filters out invalid results.

        Args:
            chunk (Dataset): A subset of the dataset.

        Returns:
            list: List of valid parsed Items.
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        
        Process dataset chunks in parallel using multiple processes.

        Args:
            workers (int): Number of worker processes.

        Returns:
            list: Aggregated list of all parsed Items.
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        
        Process dataset chunks in parallel using multiple processes.

        Args:
            workers (int): Number of worker processes.

        Returns:
            list: Aggregated list of all parsed Items.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        
        Load and process the dataset.

        Steps:
        1. Downloads dataset from Hugging Face
        2. Splits into chunks
        3. Processes in parallel
        4. Returns cleaned items

        Args:
            workers (int, optional): Number of parallel workers.
                                     Defaults to CPU count - 1.

        Returns:
            list: Processed list of Items.
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results