import os
from litellm import completion


DEFAULT_MODEL_NAME=os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
