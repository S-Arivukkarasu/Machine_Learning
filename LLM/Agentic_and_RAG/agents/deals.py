import re
import time
import requests
import feedparser
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List, Self, Dict
from pydantic import BaseModel, Field


feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


def extract(html_snippet: str) -> str:
    """
    Extract clean, readable text from an HTML snippet.

    Parses the provided HTML using BeautifulSoup, locates the deal summary
    section when available, removes HTML markup, normalizes whitespace, and
    returns the resulting text.

    Args:
        html_snippet: HTML content containing the deal summary.

    Returns:
        A cleaned plain-text representation of the deal summary.
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet

    return result.replace("\n", " ")


class ScrapedDeal:
    """
    Represent a deal retrieved from a DealNews RSS feed.

    A ScrapeDeal instance contains the deal title, summary, URL, details,
    and product features extracted from the deal's RSS entry and web page.
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Initialize a ScrapeDeal from an RSS feed entry.

        Extracts the deal title, summary, and URL from the RSS entry, then
        retrieves and parses the deal page to obtain its details and features.

        Args:
            entry: Dictionary containing the RSS feed entry data.
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        """
        Limit text fields to reasonable lengths.

        Truncates the title, details, and features fields to prevent excessive
        amounts of text from being passed to a language model.
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a concise string representation of the deal.

        Returns:
            The deal title enclosed in angle brackets.
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a detailed text representation of the deal.

        Formats the title, details, features, and URL into a single string
        suitable for providing deal information to a language model.

        Returns:
            A formatted string containing the deal information.
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool=False) -> List[Self]:
        """
        Fetch deals from all configured RSS feeds.

        Parses each RSS feed and retrieves up to the first 10 entries from
        each feed. Each entry is converted into a ScrapeDeal instance.

        Args:
            show_progress: Whether to display a progress bar while processing
                the RSS feeds. Defaults to False.

        Returns:
            A list of ScrapeDeal instances retrieved from the configured feeds.
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.05)
        return deals

    
class Deal(BaseModel):
    """
    Represent a selected deal with a concise product description.

    Stores the product description, advertised price, and URL associated
    with a deal selected from the scraped deal data.
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    Represent a collection of selected deals.

    Contains the five deals judged to have the most detailed descriptions,
    clearest pricing, and strongest overall deal information.
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    Represent a potential deal opportunity.

    Associates a selected deal with an estimated value and calculates the
    discount relative to that estimated value.
    """

    deal: Deal
    estimate: float
    discount: float