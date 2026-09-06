import modal
from modal import Image

# Setup

app = modal.App("Hello")
image = Image.debian_slim().pip_install("requests")

# Simple Hello Program

@app.function(image=image)
def hello() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city} {region} {country}"


# You can also add region

@app.function(image=image, region='eu')
def hello_europe() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city} {region} {country}"