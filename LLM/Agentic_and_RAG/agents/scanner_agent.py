from ollama import chat
from agents.agent import Agent
from typing import Optional, List
from agents.deals import ScrapedDeal, DealSelection


class ScannerAgent(Agent):
    """
    Agent responsible for scanning newly scraped deals and selecting the most
    promising products based on description quality and price clarity.

    The scanner retrieves deals from configured RSS feeds, filters out deals
    that have already been processed, and uses an Ollama language model to
    identify the five deals with the most detailed product descriptions and
    clearly identifiable prices.

    Attributes:
        MODEL: Name of the Ollama model used to analyze scraped deals.
        SYSTEM_PROMPT: Instructions provided to the language model describing
            how deals should be evaluated and formatted.
        USER_PROMPT_PREFIX: Prefix used when constructing the user prompt.
        USER_PROMPT_SUFFIX: Suffix requiring exactly five selected deals.
        name: Human-readable name of the agent.
        color: Display color inherited from the Agent base class.
    """

    MODEL = "qwen3"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    """

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Initialize the ScannerAgent.

        Logs messages indicating that the scanner agent is being initialized
        and is ready for use. No external client or additional state is
        required because Ollama is invoked directly when a scan is performed.
        """
        self.log("Scanning Agent Initializing")
        self.log("Scanner Agent is Ready")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        """
        Fetch new deals from the configured RSS feeds.

        Retrieves scraped deals using ``ScrapedDeal.fetch()`` and filters out
        any deals whose URLs already exist in the supplied memory. This
        prevents previously processed deals from being sent to the language
        model again.

        Args:
            memory: A collection of previously processed deal objects. Each
                object is expected to expose a ``deal.url`` attribute.

        Returns:
            A list of newly discovered ``ScrapedDeal`` objects that are not
            already present in ``memory``.
        """
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scrapped = ScrapedDeal.fetch()
        results = [scrape for scrape in scrapped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(results)} deals not already scrapped")
        return results

    def make_user_prompt(self, scraped) -> str:
        """
        Build the prompt used to evaluate scraped deals.

        Converts each scraped deal into a textual description and combines
        those descriptions with the predefined prompt instructions. The
        resulting prompt asks the language model to select exactly five deals
        with detailed product descriptions and clear, positive prices.

        Args:
            scraped: An iterable of ``ScrapedDeal`` objects to include in the
                prompt.

        Returns:
            A formatted string containing the selection instructions and all
            supplied deal descriptions.
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        """
        Analyze newly scraped deals and select the most promising products.

        Fetches deals that have not already been processed, sends them to the
        configured Ollama model using structured JSON output, validates the
        response against the ``DealSelection`` schema, and removes any deals
        whose resulting price is not greater than zero.

        Args:
            memory: A list or collection representing deals that have already
                been processed. Used by ``fetch_deals`` to avoid duplicate
                deals.

        Returns:
            A ``DealSelection`` containing the selected deals with a positive
            price, or ``None`` if no new deals are available.

        Raises:
            pydantic.ValidationError: If the model returns JSON that does not
                conform to the ``DealSelection`` schema.
        """
        scraped = self.fetch_deals(memory=memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling Ollama using Structured Outputs")
            result = chat(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=DealSelection.model_json_schema(),
                think=False,
                options={
                    "temperature": 0,
                },
            )
            result = DealSelection.model_validate_json(result.message.content)
            result.deals = [deal for deal in result.deals if deal.price>0]
            self.log(
                f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI"
            )
            return result
        return None

    def test_scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        """
        Return a predefined set of deals for testing.

        Provides a static ``DealSelection`` containing representative products
        with detailed descriptions, prices, and URLs. This method avoids
        fetching live RSS data or calling the language model, making it useful
        for unit tests, development, and demonstrations.

        Args:
            memory: Accepted for API compatibility with ``scan`` but is not
                currently used.

        Returns:
            A ``DealSelection`` containing predefined test deals.
        """
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers stunning picture quality with 3840x2160 resolution. It features Dolby Vision HDR and HDR10 compatibility, ensuring a vibrant and dynamic viewing experience. The TV runs on Roku's operating system, allowing easy access to streaming services and voice control compatibility with Google Assistant and Alexa. With three HDMI ports available, connecting multiple devices is simple and efficient.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display designed specifically for remote work and video conferencing. With a native resolution of 1080p, it provides crystal-clear video quality, featuring a privacy shutter and stereo speakers. This display includes a 1080p webcam with manual pan, tilt, and zoom control, along with an ambient light sensor to adjust the vanity lighting as needed. It also supports 5W wireless charging for mobile devices, making it an all-in-one solution for home offices.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by a 7th generation AMD Ryzen 5 8645HS 6-core CPU, offering efficient performance for multitasking and demanding applications. It features a 16-inch touch display with a resolution of 1920x1080, ensuring bright and vivid visuals. Accompanied by 16GB of RAM and a 512GB SSD, the laptop provides ample speed and storage for all your files. This model is designed to handle everyday tasks with ease while delivering an enjoyable user experience.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop is equipped with a 6th-generation AMD Ryzen 5 7640HS 6-Core CPU, providing powerful performance for gaming and content creation. It features a 15.6-inch 1080p display with a 120Hz refresh rate, allowing for smooth and responsive gameplay. With 16GB of RAM and a substantial 1TB NVMe M.2 SSD, this laptop ensures speedy performance and plenty of storage for games and applications. Additionally, it includes the Nvidia GeForce RTX 3050 GPU for enhanced graphics and gaming experiences.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        return DealSelection(**results)