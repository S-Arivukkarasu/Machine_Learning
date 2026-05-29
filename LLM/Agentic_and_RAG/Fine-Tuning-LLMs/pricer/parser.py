from pricer.items import Item
import json
import re


# Minimum number of characters required for the product text
MIN_CHARS = 600
# Minimum allowed price for items
MIN_PRICE = 0.5
# Maximum allowed price for items
MAX_PRICE = 999.49
# Maximum characters allowed for each text field
MAX_TEXT_EACH = 3000
# Maximum characters allowed for the combined text
MAX_TEXT_TOTAL = 4000

# Fields in product metadata that should be removed
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

def simplify(text_list) -> str:
    """
    Return a simplified string without too much whitespace and limited to MAX_TEXT characters
    Clean and simplify text fields.

    This function removes unnecessary whitespace characters and
    truncates the resulting text to a maximum allowed length.

    Args:
        text_list (list | str):
            Text data such as product descriptions or features.

    Returns:
        str:
            A cleaned and simplified string limited to MAX_TEXT_EACH characters.
    """
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )

    
def scrub(title, description, features, details) -> str:
    """
    Return a cleansed full string with product numbers and unimportant details removed
    Create a cleaned product text representation.

    This function:
    - Removes unnecessary metadata fields
    - Combines title, description, features, and details
    - Removes product codes and serial numbers
    - Limits the final text size

    Args:
        title (str):
            Product title.
        description (str | list):
            Product description text.
        features (list | str):
            Product feature list.
        details (dict):
            Additional product metadata.

    Returns:
        str:
            A cleaned text block representing the product,
            limited to MAX_TEXT_TOTAL characters.
    """
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details):
    """
    Extract and normalize item weight from product details.

    Converts weight values from different units into pounds.

    Supported units:
    - pounds
    - ounces
    - grams
    - milligrams
    - kilograms
    - hundredths of pounds

    Args:
        details (dict):
            Product metadata containing weight information.

    Returns:
        float:
            Weight in pounds. Returns 0 if weight is not found.
    """
    weight_str = details.get("Item Weight")
    if weight_str:
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit == "pounds":
            return amount
        elif unit == "ounces":
            return amount / 16
        elif unit == "grams":
            return amount / 453.592
        elif unit == "milligrams":
            return amount / 453592
        elif unit == "kilograms":
            return amount / 0.453592
        elif unit == "hundredths" and parts[2].lower() == "pounds":
            return amount / 100
    return 0


def parse(datapoint, category):
    """
    Convert a raw dataset entry into an Item object.

    This function validates product data by:
    - Ensuring the price is within an acceptable range
    - Cleaning product text
    - Extracting product weight
    - Checking minimum text length

    If the datapoint meets all requirements, an Item
    instance is created.

    Args:
        datapoint (dict):
            Raw product data from the dataset.
        category (str):
            Category label assigned to the product.

    Returns:
        Item | None:
            Returns an Item object if the datapoint is valid.
            Returns None if the datapoint fails validation.
    """
    try:
        price = float(datapoint["price"])
    except ValueError:
        return None
    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = json.loads(datapoint["details"])
        weight = get_weight(details)
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:
            return Item(
                title=title,
                category=category,
                price=price,
                full=full,
                weight=weight,
            )