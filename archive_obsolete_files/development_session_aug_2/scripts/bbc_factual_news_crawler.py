import asyncio
import csv
import random
import time
from datetime import datetime
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, URLPatternFilter
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

OUTPUT_CSV = "bbc_factual_news.csv"
RECENT_DAYS = 30
MAX_PAGES = 1000
MAX_DEPTH = 10

# Factual news categories to include
INCLUDE_PATTERNS = [
    "/news/world*", "/news/uk*", "/news/business*", "/news/science*", "/news/technology*", "/news/health*", "/news/politics*"
]
# Exclude opinion, sport, entertainment, lifestyle, old, and non-factual
EXCLUDE_PATTERNS = [
    "/sport/", "/entertainment/", "/lifestyle/", "/celebrity/", "/influencer/", "/live/", "/av/", "/video/", "/gallery/", "/archive/", "/magazine/", "/weather/", "/travel/", "/food/", "/recipes/", "/culture/", "/arts/", "/music/", "/fashion/", "/history/", "/education/", "/learning/", "/kids/", "/children/", "/cbebies/", "/cbbc/", "/iplayer/", "/sounds/", "/radio/", "/podcasts/", "/bitesize/", "/shop/", "/events/", "/tickets/", "/games/", "/quiz/", "/puzzles/", "/opinion/", "/comment/", "/analysis/", "/blog/", "/editorial/", "/columns/", "/columnists/", "/features/", "/specials/", "/in-pictures/", "/inpictures/", "/picture/", "/photo/", "/photos/", "/galleries/", "/old/", "/2019/", "/2018/", "/2017/", "/2016/", "/2015/", "/2014/", "/2013/", "/2012/", "/2011/", "/2010/", "/2009/", "/2008/", "/2007/", "/2006/", "/2005/", "/2004/", "/2003/", "/2002/", "/2001/", "/2000/", "/health/"
]

# Helper to parse date from BBC article metadata (if available)
def parse_bbc_date(meta):
    for key in ["datePublished", "date", "article:published_time"]:
        if key in meta:
            try:
                return datetime.fromisoformat(meta[key][:19])
            except Exception:
                continue
    return None



async def crawl_bbc_factual_news():
    filter_chain = FilterChain([
        DomainFilter(allowed_domains=["bbc.co.uk"]),
        URLPatternFilter(patterns=INCLUDE_PATTERNS)
    ])
    strategy = BestFirstCrawlingStrategy(
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES,
        filter_chain=filter_chain
    )
    # Note: Crawl4AI version does not support headers in CrawlerRunConfig. User-Agent randomization not available here.
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        stream=True
    )
    results = []
    seen_urls = set()
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun("https://www.bbc.co.uk/news", config=config):
            if not result.success:
                print(f"[SKIP] {getattr(result, 'url', 'NO_URL')} | Reason: crawl unsuccessful")
                continue
            url = result.url
            # Manual exclusion for unwanted patterns
            if any(excl in url for excl in EXCLUDE_PATTERNS):
                print(f"[SKIP] {url} | Reason: excluded by EXCLUDE_PATTERNS")
                continue
            if url in seen_urls:
                print(f"[SKIP] {url} | Reason: duplicate URL")
                continue
            seen_urls.add(url)
            meta = getattr(result, 'metadata', {}) or {}
            title = getattr(result, 'title', None) or meta.get('title', '')
            content = getattr(result, 'content', None) or meta.get('articleBody', '')
            date = parse_bbc_date(meta)
            # Date restriction removed: allow any date
            # Skip if content is too short or missing
            if not content:
                print(f"[SKIP] {url} | Reason: no content extracted")
                continue
            if len(content) < 500:
                print(f"[SKIP] {url} | Reason: content too short (len={len(content)})")
                continue
            # Extract category from URL
            category = None
            for cat in ["world", "uk", "business", "science", "technology", "health", "politics"]:
                if f"/news/{cat}" in url:
                    category = cat
                    break
            results.append({
                "url": url,
                "title": title,
                "content": content,
                "date": date.isoformat() if date else None,
                "category": category
            })
            print(f"[FOUND] {title} | {category} | {date} | {url}")
            # Human-like delay
            time.sleep(random.uniform(1.5, 4.0))
    # Write to CSV
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["url", "title", "content", "date", "category"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nDone! {len(results)} factual news articles saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(crawl_bbc_factual_news())
