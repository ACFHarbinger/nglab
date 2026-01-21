"""
News Crawler for Financial Sentiment Analysis.

Fetches data from financial news RSS feeds and websites.
"""

from typing import Any, ClassVar

import feedparser


class NewsCrawler:
    """
    Crawler for financial news feeds.
    """

    DEFAULT_FEEDS: ClassVar[list[str]] = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC",  # S&P 500
        "https://www.marketwatch.com/rss/topstories",
        "https://search.cnbc.com/rs/search/view.xml?partnerId=2000&keywords=finance",
    ]

    def __init__(self, feeds: list[str] | None = None) -> None:
        """Initialize NewsCrawler."""
        self.feeds = feeds or self.DEFAULT_FEEDS

    def crawl(self) -> list[dict[str, Any]]:
        """
        Crawl RSS feeds and return a list of news items.

        Returns:
            List of dictionaries with 'title', 'summary', 'link', 'published'.
        """
        all_news: list[dict[str, Any]] = []
        for url in self.feeds:
            # Note: In a real production environment, we should handle timeouts and retries.
            feed = feedparser.parse(url)
            for entry in feed.entries:
                all_news.append(
                    {
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": url,
                    }
                )

        return all_news


def main_crawler(url: str | None = None) -> list[dict[str, Any]]:
    """Entry point for the webcrawler command."""
    feeds = [url] if url else None
    crawler = NewsCrawler(feeds)
    print(f"Crawling {len(crawler.feeds)} feeds...")
    news = crawler.crawl()
    print(f"Found {len(news)} news items.")

    # Showcase some results
    for item in news[:5]:
        print(f"- {item['title']} ({item['published']})")

    return news
