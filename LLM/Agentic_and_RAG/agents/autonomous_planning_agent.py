import os
import json
import litellm
from agents.agent import Agent
from typing import Optional, List, Dict
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
# from agents.messaging_agent import MessagingAgent


class AutonomousPlanningAgent(Agent):
    """
    Orchestrates the bargain-finding workflow by coordinating the scanner and
    ensemble valuation agents.

    The agent uses an LLM with tool-calling capabilities to:
    1. Scan the internet for potential bargain deals.
    2. Estimate the true value of each discovered item.
    3. Select the most compelling opportunity based on the price-to-value gap.
    4. Optionally notify the user about the selected opportunity.

    The agent maintains the current scan memory and selected opportunity during
    a planning run. It also configures LiteLLM to communicate with the
    configured model endpoint.

    Attributes:
        name: Human-readable name of the agent.
        color: Display color inherited from the base Agent class.
        MODEL: Model identifier used for LLM-based planning.
        scanner: Agent responsible for finding bargain deals.
        ensemble: Agent responsible for estimating the true value of products.
        memory: Previously surfaced deal URLs used to avoid duplicate results.
        opportunity: The best opportunity identified during the current run.
        base_url: Endpoint URL used to access the configured model.
        token_id: Authentication token identifier for the model endpoint.
        token_secret: Authentication secret for the model endpoint.
    """
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "openai/Qwen/Qwen3.5-35B-A3B-FP8"

    def __init__(self, collection):
        """
        Initialize the autonomous planning agent and its supporting agents.

        Args:
            collection: Collection or data source passed to the
                EnsembleAgent for product valuation.
        """
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        # self.messaging = MessagingAgent()
        self.memory = None
        self.opportunity = None
        self.base_url = os.environ['MODAL_ENDPOINT_URL']
        self.token_id = os.environ['MODAL_PROXY_TOKEN_ID']
        self.token_secret = os.environ['MODAL_PROXY_TOKEN_SECRET']
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        """
        Scan the internet for bargain products using the ScannerAgent.

        The current memory is passed to the scanner so previously surfaced
        deals can be taken into account. The scanner's results are serialized
        to JSON for consumption by the LLM tool-calling workflow.

        Returns:
            A JSON string containing the discovered bargain deals, or
            ``"No Deals Found"`` when the scanner returns no results.
        """
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No Deals Found"

    def estimate_true_value(self, description: str) -> str:
        """
        Estimate the actual market value of a product.

        Args:
            description: Textual description of the product whose true value
                should be estimated.

        Returns:
            A human-readable string containing the product description and
            its estimated true value.
        """
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"

    # def notify_user_of_deal(
    #     self, description: str, deal_price: float, estimated_true_value: float, url: str
    # ) -> Dict:
    #     """
    #     Run the tool to notify the user
    #     """
    #     if self.opportunity:
    #         self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
    #     else:
    #         self.log("Autonomous Planning agent is notifying user")
    #         self.messenger.notify(description, deal_price, estimated_true_value, url)
    #         deal = Deal(product_description=description, price=deal_price, url=url)
    #         discount = estimated_true_value - deal_price
    #         self.opportunity = Opportunity(
    #             deal=deal, estimate=estimated_true_value, discount=discount
    #         )
    #     return "Notification sent ok"

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal scraped from the internet",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value that this is worth",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this deal as scraped from the internet",
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the tool definitions available to the planning LLM.

        The tools allow the LLM to scan for bargains and estimate the true
        value of discovered products.

        Returns:
            A list of OpenAI-compatible function tool definitions.
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            # {"type": "function", "function": self.notify_function},
        ]
    
    def handle_tool_call(self, message):
        """
        Execute the tools requested by an LLM response.

        Each tool call in the message is mapped to its corresponding Python
        method, its JSON arguments are decoded, and the method is executed.
        The results are returned in the format expected by the LLM for
        subsequent tool-calling iterations.

        Args:
            message: An LLM response message containing one or more tool calls.

        Returns:
            A list of tool-result messages containing the output of each
            executed tool.
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            # "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name 
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Execute the complete autonomous bargain-discovery workflow.

        The LLM is instructed to scan for bargains, estimate the true value
        of each discovered product, and identify the most compelling deal.
        Tool calls are executed iteratively until the LLM completes its
        response.

        Args:
            memory: A list of URLs representing deals that have already been
                surfaced in previous runs. These are passed to the scanner to
                help avoid resurfacing the same opportunities.

        Returns:
            The selected ``Opportunity`` if a deal was identified and
            surfaced; otherwise, ``None``.
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]
        done = False
        litellm.api_base = (self.base_url)
        litellm.api_key = (
            f"{self.token_id}."
            f"{self.token_secret}"
        )

        while not done:
            response = litellm.completion(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True

        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity