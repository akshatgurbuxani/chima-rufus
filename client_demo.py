from Rufus.client import RufusClient
import os 

key = os.getenv('Rufus_API_KEY')
client = RufusClient(api_key=key)

instructions = "Find information about product features and customer FAQs."
documents = client.extract_data("https://example.com", instructions=instructions, output_file="demo_client_out.json")

print(f"Successfully scraped the websites! Documents are saved to the demo_client_out.json file.")