import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import regex as re
import json

class AsyncCrawler:
    def __init__(self, max_depth=3, max_workers=5):
        self.max_depth = max_depth
        self.visited_urls = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def fetch(self, session, url):
        try:
            async with session.get(url) as response:
                return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def extract_links(self, session, url):
        html = await self.fetch(session, url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        return list(set(links))

    async def extract_content(self, session, url):
        html = await self.fetch(session, url)
        if not html:
            return {}
        
        soup = BeautifulSoup(html, 'html.parser')
        content = {
            "url": url,
            "title": soup.title.string.strip() if soup.title else "",
            "headings": [h.get_text(strip=True) for h in soup.find_all(re.compile('^h[1-6]$'))],
            "text": [p.get_text(strip=True) for p in soup.find_all('p')]
        }
        return content

    async def crawl(self, url, depth=0):
        if depth > self.max_depth or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        async with aiohttp.ClientSession() as session:
            links = await self.extract_links(session, url)
            content = await self.extract_content(session, url)
            results = [content]
            
            tasks = [self.crawl(link, depth + 1) for link in links if urlparse(url).netloc == urlparse(link).netloc]
            nested_results = await asyncio.gather(*tasks)
            
            for res in nested_results:
                results.extend(res)
            
            return results

    def run(self, start_url):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.crawl(start_url))
    

if __name__ == "__main__":
    crawler = AsyncCrawler(max_depth=2, max_workers=5)
    results = crawler.run("https://www.example.com")
    with open("crawled_data.json", "w") as f:
        json.dump(results, f, indent=4)
    # print(results)