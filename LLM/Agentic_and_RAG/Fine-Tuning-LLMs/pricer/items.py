from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price

    Represents a single product item with pricing information.

    This model stores product attributes such as title, category,
    price, description, and prompt used for language model training.
    It also provides utility methods for generating prompts and
    uploading/downloading datasets from the Hugging Face Hub.
    """

    # Variables (Attributes)
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        """TGenerate a training prompt for a language model.

        The prompt consists of:
        1. A predefined question asking the price
        2. The product description text
        3. The price formatted to the nearest dollar

        Args:
            text (str): Product description used in the prompt.

        Returns:
            None: The generated prompt is stored in `self.prompt`.
        """
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """
        Generate a test prompt without the actual price.

        This is used during evaluation so that the model
        predicts the price itself.

        Returns:
            str: Prompt ending with the prefix where the
            model should generate the price.
        """
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        """
        Return a readable string representation of the Item.

        Useful for debugging and printing objects.

        Returns:
            str: A string showing the product title and price.
        """
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """
        Push Item lists to HuggingFace Hub.
        
        Upload Item datasets to the Hugging Face Hub.

        Converts Item objects into dictionaries and organizes them
        into train, validation, and test splits before pushing them
        to the hub.

        Args:
            dataset_name (str): Name of the dataset repository on Hugging Face.
            train (list[Item]): List of training items.
            val (list[Item]): List of validation items.
            test (list[Item]): List of test items.

        Returns:
            None
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """
        Load from HuggingFace Hub and reconstruct Items

        Load an Item dataset from the Hugging Face Hub.

        Downloads the dataset and reconstructs Item objects
        from the stored dictionary rows.

        Args:
            dataset_name (str): Name of the dataset repository on Hugging Face.

        Returns:
            tuple:
                - List of training Item objects
                - List of validation Item objects
                - List of test Item objects
        """
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
        

    def count_tokens(self, tokenizer):
        """
        Count tokens in the item summary.
    
        Encodes the summary text using the provided tokenizer
        and returns the total number of tokens. This is useful
        for filtering, truncating, or analyzing dataset entries
        before training.
    
        Args:
            tokenizer: Tokenizer used to encode the summary text.
    
        Returns:
            int: Number of tokens in the summary.
        """
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens, do_round):
        """
        Generate prompt and completion fields for training.
    
        Creates a prompt-completion pair from the item summary
        and price. If the summary exceeds the specified token
        limit, it is truncated to the maximum allowed number
        of tokens before being used in the prompt.
    
        The generated prompt consists of:
        1. A predefined pricing question
        2. The product summary
        3. A price prefix indicating where the model should
           generate the answer
    
        The completion contains the product price, optionally
        rounded to the nearest dollar.
    
        Args:
            tokenizer: Tokenizer used to count and truncate
                summary tokens.
            max_tokens (int): Maximum number of tokens allowed
                in the summary portion of the prompt.
            do_round (bool): If True, round the price to the
                nearest dollar and format it as ``XX.00``.
                Otherwise, use the original price value.
    
        Returns:
            None: The generated values are stored in
                ``self.prompt`` and ``self.completion``.
        """
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer):
        """
        Count tokens in the full prompt-completion pair.
    
        Combines the generated prompt and completion text,
        tokenizes the result, and returns the total number
        of tokens. This is useful for estimating training
        costs and ensuring examples fit within model context
        limits.
    
        Args:
            tokenizer: Tokenizer used to encode the text.
    
        Returns:
            int: Total number of tokens in the concatenated
                prompt and completion.
        """
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    def to_datapoint(self) -> dict:
        """
        Convert the Item into a prompt-completion datapoint.
    
        Creates a dictionary containing only the fields
        required for supervised fine-tuning (SFT), namely
        the prompt and its corresponding completion.
    
        Returns:
            dict: A dictionary with the following keys:
                - "prompt": Input text provided to the model.
                - "completion": Expected model output.
        """
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        """
        Push prompt-completion datasets to the Hugging Face Hub.
    
        Converts Item objects into prompt-completion dictionaries
        using ``to_datapoint()`` and organizes them into training,
        validation, and test splits. The resulting dataset is then
        uploaded to the specified Hugging Face Hub repository for
        supervised fine-tuning (SFT).
    
        Args:
            dataset_name (str): Name of the dataset repository on
                Hugging Face.
            train (list[Item]): Training examples.
            val (list[Item]): Validation examples.
            test (list[Item]): Test examples.
    
        Returns:
            None
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)
