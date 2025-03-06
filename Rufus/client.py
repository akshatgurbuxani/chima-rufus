import os
import json
from .instruction_parser import InstructionParser
from .crawler import AsyncCrawler

class RufusClient:
    """
    This client utilizes web crawling and AI-driven instruction parsing to gather and structure data 
    from web pages, accommodating dynamic content and user-defined extraction criteria.

    Attributes:
        api_key (str): The OpenAI API key for parsing instructions and extracting keywords.
        instruction_parser (InstructionParser): Responsible for filtering content based on user instructions.
        web_crawler (WebCrawler): A web crawler that extracts content from specified URLs.

    Methods:
        extract_data(url, instructions, output_file): Extracts data from a website based on the 
                                                      provided URL and instructions, saving results 
                                                      to a JSON file or returning structured documents.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.instruction_parser = InstructionParser(self.api_key)
        self.web_crawler = AsyncCrawler()

    def extract_data(self, url, instructions, output_file=None):
        """
        Extracts data from the specified URL based on user instructions.

        :param url: The URL of the website to scrape.
        :param instructions: User-defined instructions for filtering the content.
        :param output_file: Optional filename to save the results in JSON format.
        :return: Structured data extracted from the website.
        """
        crawled_data = self.web_crawler.run(url)  # Get raw crawled data

        # Filter the crawled data based on user instructions
        filtered_data = self.instruction_parser.filter_content(crawled_data, instructions)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(filtered_data, f, indent=4)

        return filtered_data