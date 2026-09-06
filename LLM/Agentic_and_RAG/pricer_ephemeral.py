import modal
from modal import Image

# Setup

app = modal.App("Price Large and Lite")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
    )
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "Amazon-price-predictor-lite"
HF_USER = "Arivukkarasu"
REVISION = "19708427f0618548aaa8a94b72d06ad741769d66"
REVISION_LARGE = "ebc87bc927145328e0d70f2946669cc8856b00d8"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_NAME}"
FINETUNED_MODEL_LARGE = f"{HF_USER}/Amazon-price-predictor-2026-07-06_07.24.32"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1200)
def price(description: str) -> str:
    import re
    import torch 
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    PREFIX = "Price is $"
    QUESTION = "What does this cost to the nearest dollar?"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    # Quant config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load Model ad Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )

    fine_tuned_model = PeftModel.from_pretrained(
        base_model, FINETUNED_MODEL, revision=REVISION,
    )

    set_seed(42)
    inputs=tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0])
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1200)
def price_large(description: str) -> str:
    import re
    import torch 
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    PREFIX = "Price is $"
    QUESTION = "What does this cost to the nearest dollar?"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    # Quant config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load Model ad Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )

    fine_tuned_model = PeftModel.from_pretrained(
        base_model, FINETUNED_MODEL_LARGE, revision=REVISION_LARGE,
    )

    set_seed(42)
    inputs=tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0])
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
