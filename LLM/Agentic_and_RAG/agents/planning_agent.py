from typing import Optional, List

from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
# from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):
    """
    Coordinates the deal-scanning and pricing workflow.

    The PlanningAgent orchestrates the ScannerAgent and EnsembleAgent to
    identify potential deals, estimate their market value, calculate the
    potential discount, and determine whether a deal meets the configured
    threshold for an opportunity.

    Attributes:
        name: Display name of the agent.
        color: Console/logging color associated with the agent.
        DEAL_THRESHOLD: Minimum discount required for a deal to qualify
            as an opportunity.
        scanner: Agent responsible for scanning available deal sources.
        ensemble: Agent responsible for estimating product prices.
    """
    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Initialize the PlanningAgent and its dependent agents.

        Args:
            collection: Collection or data source used by the
                EnsembleAgent for product price estimation.
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        # self.message = MessagingAgent()
        self.log("Planning Agent is Ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Evaluate a deal and calculate its potential discount.

        The deal's product description is passed to the EnsembleAgent to
        estimate its value. The estimated value is then compared with the
        deal's current price to calculate the potential discount.

        Args:
            deal: Deal containing the product description and current price.

        Returns:
            An Opportunity containing the original deal, estimated value,
            and calculated discount.
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the complete deal discovery and evaluation workflow.

        The ScannerAgent searches for new deals while excluding URLs that
        have already been processed. Up to five discovered deals are then
        evaluated and sorted by their calculated discount. The highest
        discounted deal is returned only if its discount exceeds
        DEAL_THRESHOLD.

        Args:
            memory: List of URLs that have already been surfaced and should
                be excluded from the current scan.

        Returns:
            The best Opportunity if its discount exceeds DEAL_THRESHOLD;
            otherwise, None.
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if selection:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            if best.discount > self.DEAL_THRESHOLD:
                # self.messager.alert(best)
                pass
            self.log("Planning Agent has completed a run")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None
