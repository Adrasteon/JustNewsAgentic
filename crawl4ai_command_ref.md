
# Crawl4AI Command Reference

**Version:** 0.7.2+  
**Audience:** Copilot agents & developers (human-readable)

---

## Overview
Crawl4AI is a modular, agent-friendly web crawling and content extraction system. It supports deep crawling, intelligent filtering, browser/session management, advanced extraction, and CLI automation. This reference covers all major Python APIs, CLI commands, strategies, filters, and best practices for full system utilization.

---

## 1. Python API: Core Classes & Usage

### `AsyncWebCrawler`
- **Purpose:** Main async crawler for single/multi-page, deep, and dynamic crawling.
- **Usage:**
  ```python
  async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
      result = await crawler.arun(url, config=CrawlerRunConfig(...))
  ```
- **Key Methods:**
  - `arun(url, config)`: Crawl a single URL
  - `arun_many(urls, config)`: Crawl multiple URLs in parallel
  - Streaming: `async for result in await crawler.arun(url, config)`

### `CrawlerRunConfig`
- **Purpose:** Controls crawl behavior, extraction, filtering, browser, and more.
- **Key Parameters:**
  - `cache_mode`: `CacheMode.BYPASS`/`ENABLED`
  - `deep_crawl_strategy`: BFS/DFS/BestFirst (see below)
  - `scraping_strategy`: e.g. `LXMLWebScrapingStrategy`
  - `markdown_generator`: e.g. `DefaultMarkdownGenerator`
  - `extraction_strategy`: LLM/CSS/JSON/Custom
  - `excluded_tags`, `remove_overlay_elements`, `css_selector`, `js_code`, `session_id`, `screenshot`, `pdf`, `proxy_config`, `stream`, `verbose`, etc.

### `BrowserConfig`
- **Purpose:** Controls browser type, headless mode, viewport, proxy, JS, user agent, geolocation, etc.
- **Key Parameters:**
  - `browser_type`: "chromium" (default), "firefox", "webkit"
  - `headless`: True/False
  - `java_script_enabled`: True/False
  - `proxy_config`: {server, username, password}
  - `user_agent_mode`, `user_agent_generator_config`, `viewport_width`, `viewport_height`, etc.

### Extraction Strategies
- **LLMExtractionStrategy**: Use LLMs (OpenAI, Groq, Ollama, etc.) for schema-based or freeform extraction
- **JsonCssExtractionStrategy**: Fast, CSS selector-based structured extraction
- **CosineStrategy**: Semantic/keyword clustering and extraction

### Content Filtering
- **PruningContentFilter**: Remove low-value content (threshold, min_word_threshold)
- **BM25ContentFilter**: Relevance-based filtering

### Hooks & Customization
- **Hooks:** `set_hook("before_goto", fn)`, `set_hook("on_execution_started", fn)`
- **Custom JS:** `js_code` param or `js_only=True` for dynamic content

---

## 2. Deep Crawling: Strategies, Filters, Scorers

### Strategies
- **BFSDeepCrawlStrategy**: Breadth-first, level-by-level
- **DFSDeepCrawlStrategy**: Depth-first, as deep as possible
- **BestFirstCrawlingStrategy**: Prioritizes high-value/score pages first

#### Key Parameters (all strategies):
  - `max_depth`: How deep to crawl (0 = just root)
  - `max_pages`: Max total pages
  - `include_external`: Follow off-domain links
  - `filter_chain`: List of filters (see below)
  - `url_scorer`: Prioritize by score (e.g. `KeywordRelevanceScorer`)
  - `score_threshold`: Only crawl above this score
  - `stream`: Yield results as found

### Filters (for `FilterChain`)
- **DomainFilter**: Restrict to allowed domains
- **URLPatternFilter**: Wildcard/regex URL allow/block
- **ContentTypeFilter**: Only crawl certain MIME types
- **SEOFilter**: Filter by SEO keyword presence/score
- **ContentRelevanceFilter**: Filter by semantic similarity to a query

#### Example:
```python
filter_chain = FilterChain([
    DomainFilter(allowed_domains=["docs.crawl4ai.com"]),
    URLPatternFilter(patterns=["*core*", "*advanced*"]),
    ContentTypeFilter(allowed_types=["text/html"]),
])
strategy = BestFirstCrawlingStrategy(
    max_depth=2, max_pages=10, filter_chain=filter_chain
)
```

### Scorers
- **KeywordRelevanceScorer**: Score URLs by keyword relevance
- **Custom scorer**: Any callable returning a float

---

## 3. Extraction, Media, and Advanced Features

### Extraction
- **LLM Extraction**: Use LLMs for structured/freeform extraction (with schema, instruction, provider, etc.)
- **CSS Extraction**: Use CSS selectors for fast, schema-based extraction
- **CosineStrategy**: Cluster/semantic extraction with keyword/embedding filtering

### Media & Links
- `result.media["images"]`: List of images (src, alt, score)
- `result.links["internal"]`, `result.links["external"]`: All links

### Screenshots & PDFs
- `screenshot=True`/`pdf=True` in `CrawlerRunConfig` to capture
- `result.screenshot`/`result.pdf`: Base64-encoded data

### Raw HTML & Local Files
- `url="raw:<html>...</html>"` or `url="file:///path/to/file.html"`

### Proxy & Rotation
- `proxy_config` in `BrowserConfig` for single proxy
- `RoundRobinProxyStrategy` for rotating proxies

### Session & User Simulation
- `session_id` for persistent sessions
- `simulate_user`, `override_navigator`, `user_agent_mode` for anti-bot

### SSL Certificate Extraction
- `fetch_ssl_certificate=True` in `CrawlerRunConfig`
- `result.ssl_certificate`: Info, export to PEM/DER/JSON

---

## 4. CLI Usage: `crwl` Command

### Main Commands
| Command | Inputs / flags | What it does |
|---|---|---|
| **profiles** | *(none)* | Manage browser profiles (list, create, delete, use) |
| **browser status** | – | Show builtin browser status |
| **browser stop** | – | Kill builtin browser |
| **browser view** | `--url, -u` | Open browser window to URL |
| **config list/get/set** | `key`, `key value` | Manage global config |
| **examples** | – | Show CLI usage samples |
| **crawl** | `url`, `--browser-config`, `--crawler-config`, `--filter-config`, `--extraction-config`, `--json-extract`, `--schema`, `--browser`, `--crawler`, `--output`, `--output-file`, `--bypass-cache`, `--question`, `--verbose`, `--profile` | One-shot crawl + extraction |
| **(default)** | Same as crawl | Shortcut: `crwl <url>` |

#### Example:
```bash
crwl https://site.com -p my-profile -o json -v
crwl crawl https://site.com -b "headless=true,viewport_width=1680"
crwl https://site.com -B browser.yaml -C crawler.yaml -o markdown-fit
```

---

## 5. Browser & Profile Management

### ManagedBrowser & BrowserManager
- **Profile creation:** Save login/cookies for reuse
- **Session management:** Persistent sessions, context reuse
- **Headless/visible:** Control browser mode
- **Stealth:** Anti-bot, user agent, timezone, locale, geolocation

### Profile CLI Cheatsheet
| Scenario | Command |
|---|---|
| Launch Profile Manager | `crwl profiles` |
| Create profile | `crwl profiles` → 2 → name → login → q |
| List profiles | `crwl profiles` → 1 |
| Delete profile | `crwl profiles` → 3 |
| Crawl with profile | `crwl <url> -p my-profile` |
| Crawl with extra browser tweaks | `crwl <url> -p my-profile -b "headless=true,viewport_width=1680"` |

---

## 6. Result Object Structure

Each crawl result contains:
- `url`, `title`, `cleaned_html`, `markdown`, `raw_html`, `word_count`, `media`, `links`, `success`, `metadata`, `scout_score`, `scout_analysis`, `ssl_certificate`, `screenshot`, `pdf`, `extracted_content`

---

## 7. Best Practices & Advanced Patterns

- Use `cleaned_html` for news/article extraction (removes navigation/ads)
- Always set `filter_chain` to avoid crawling irrelevant content
- Use `max_depth` and `max_pages` to control crawl scope
- Enable `analyze_content` or LLM extraction for quality/bias/structure
- Use streaming for large/deep crawls
- Use browser profiles for authenticated/identity-based crawling
- Use session_id for persistent state across requests
- Use proxy rotation for anti-blocking
- Use hooks and custom JS for dynamic/interactive sites
- Use CLI for automation, scripting, and integration

---

## 8. Error Handling
- 400: Invalid parameters (check URL, strategy, filters)
- 500: Internal error (check logs, service health)
- Use `/health` to verify service is running

---

## 9. References
- [Crawl4AI GitHub](https://github.com/adrasteon/crawl4ai)
- [Crawl4AI API Docs](https://github.com/adrasteon/crawl4ai#api)
- [Crawl4AI Examples](https://github.com/adrasteon/crawl4ai#examples)

---

**For Copilot agents:** Always check for the latest strategies, filters, and CLI options in the Crawl4AI repo. Use `cleaned_html`, LLM/CSS extraction, and browser/session management for best results.
