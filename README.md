# Data Extraction and Filtering Project

## Overview

Rufus is an advanced web crawling tool designed to intelligently navigate and extract highly relevant content based on user input. Unlike traditional crawlers, Rufus performs deep crawling, following links across multiple layers to retrieve comprehensive information while filtering out irrelevant data. Its primary goal is to generate structured documents that can be seamlessly integrated into RAG pipelines, enabling efficient retrieval-augmented generation.

With Rufus, engineers can provide a concise prompt defining the required data, and the tool will automatically conduct a deep crawl, parsing, extracting, and synthesizing relevant content from various sources. By prioritizing high-value information and structuring it for easy processing, Rufus streamlines the data acquisition process, making it faster and more efficient for AI-driven applications.


## Project Structure

```      
├── rufus/                    # Rufus package
│   ├── client.py             # RufusClient class (main entry point)
│   ├── crawler.py            # Web crawling functionality
│   └── instructon_parser.py  # Instruction parser for AI-based content filtering
├── README.md                 # Overview and instructions
├── client_demo.py            # Rufus in action
├── client_RAG.py             # RAG pipeline demo
├── requirements.txt          # Dependenices
```


## File Structure

- `client.py`: Contains the `RufusClient` class, which orchestrates the web scraping and content filtering process.
- `instruction_parser.py`: Implements the `InstructionParser` class, responsible for filtering content based on user instructions using the gpt-3.5-turbo model.
- `crawler.py`: Contains the `AsyncCrawler` class, which handles the web scraping logic to extract content from specified URLs.
- `requirements.txt`: Lists the required Python packages for the project.
- `client_demo.py`: Shows the implementation of how a client would run the project in real time.
- `client_RAG.py`: Visualises how `RufusClient` can be used in downstream tasks like RAG pipelines.

## Detailed Rufus Component Descriptions

### 1. `client.py`

This file defines the `RufusClient` class, which is the main entry point for the application. It is responsible for:

- Initializing the OpenAI API key and the necessary components for web crawling and content filtering.
- Crawling a specified URL to extract raw content.
- Filtering the extracted content based on user instructions using the `InstructionParser`.

**Key Methods:**
- `extract_data(url, instructions, output_file)`: Scrapes data from the specified URL and filters it based on user instructions. Optionally saves the results to a JSON file.

### 2. `instruction_parser.py`

This file implements the `InstructionParser` class, which processes user instructions to filter content. It uses the OpenAI gpt-3.5-turbo model to understand the instructions and filter the provided content accordingly.

**Key Methods:**
- `filter_content(content, instructions)`: Filters the provided content based on user instructions, handling large content by splitting it into manageable chunks and calling the model for each chunk.
- `_split_content(content)`: Splits the raw content into smaller chunks based on a maximum token limit to ensure API calls do not exceed the token limit.

### 3. `crawler.py`

This file contains the `AsyncCrawler` class, which is responsible for scraping web pages. It extracts content from specified URLs using `BeautifulSoup` and returns it in a structured format (e.g., JSON).

### 2. `client_demo.py`

This file serves as a demo script to run the application. It demonstrates how to use the `DataExtractorClient` to extract and filter data from a specified URL based on user instructions.

## Demo workflow:

```python
from Rufus import RufusClient
import os 

key = os.getenv('Rufus_API_KEY')
client = RufusClient(api_key=key)

instructions = "Find information about product features and customer FAQs."
documents = client.extract_data("https://example.com", instructions=instructions, output_filename="demo_client_out.json")

print(f"Successfully scraped the websites! Documents are saved to the demo_client_out.json file.")
```

## Project Flow

1. **Setup**: Ensure you have Python installed on your machine. Create a virtual environment and install the required packages using the `requirements.txt` file.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. **Obtain OpenAI API Key**: Sign up for OpenAI and obtain an API key. Set the API key as an environment variable or pass it directly when initializing the `RufusClient`.

   ```bash
   export OPENAI_API_KEY='your_api_key'  # On Windows use `set OPENAI_API_KEY='your_api_key'`
   ```

3. **Run the Demo Script**: Use the `client_demo.py` script to extract and filter data. You can run the following command in your terminal:

   ```bash
   python client_demo.py
   ```

   Ensure that you have modified the `client_demo.py` file to include the desired URL and user instructions for filtering.
   
4. **Output**: The filtered content will be saved to `filtered_data.json` (if specified) and printed to the console.


## RAG Workflow

### 3. `client_RAG.py`

This file implements a basic RAG (Retrieval-Augmented Generation) system using LangChain and OpenAI. It scrapes a webpage, splits the content into chunks, embeds them in a FAISS vector store, and sets up a QA chain for answering queries.

**Implementation:**

```python
import os
from Rufus import RufusClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def main():
    """
    This script implements a basic RAG (Retrieval-Augmented Generation) system using LangChain and OpenAI. 
    It scrapes a webpage, splits the content into chunks, embeds them in a FAISS vector store, and sets up a QA chain for answering queries.

    Steps:

    Crawl a webpage.
    Split content into chunks.
    Embed and store chunks in FAISS.
    Set up a QA system with ChatOpenAI LLM.
    Query the system and retrieve results.
    It demonstrates using LangChain's OpenAI API for document retrieval and Q&A.
    """

    # Initialize Rufus client
    key = os.getenv('Rufus_API_KEY')
    client = RufusClient(api_key=key)

    # Rufus Crawling
    url = "https://example.com"
    instructions = "Find information about product features and customer FAQs."
    documents = client.extract_data(url=url, instructions=instructions, max_depth=3)

    # Combine all content into a single string
    text = "\n".join(documents)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
    )

    # Split the text content into chunks
    chunks = text_splitter.split_text(text)

    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    
    db = FAISS.from_documents(chunks, embeddings)

    # Create a RetrievalQA chain
    llm = ChatOpenAI(temperature=0)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Query the QA chain
    query = "User defined query"

    try:
        result = qa_chain.invoke({"query": query})
        answer = result['result']
        print("\nAnswer:", answer)
        
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")

# Run the main function
if __name__ == "__main__":
    main()
```

### Explanation of RAG Workflow Code

1. **Initialization**: The script initializes the `RufusClient` to scrape data from a specified URL.

2. **Crawling**: It uses the `extract_data` method to retrieve documents from the webpage based on user instructions.

3. **Text Processing**: The retrieved documents are combined into a single string and split into manageable chunks using `RecursiveCharacterTextSplitter`.

4. **Embedding**: The chunks are embedded using `HuggingFaceEmbeddings`, which prepares them for storage in a FAISS vector store.

5. **RetrievalQA Setup**: A `RetrievalQA` chain is created using the embedded chunks and the OpenAI model, allowing for question-answering capabilities.

6. **Querying**: The user can define a query, and the system retrieves the answer based on the embedded content.


## Challenges and Solutions

1. **Handling Large Content**: One of the main challenges was dealing with large JSON content that could exceed the token limit for API calls. This was addressed by implementing a chunking mechanism in the `InstructionParser` class, which splits the content into smaller, manageable pieces before processing.

2. **Deep Crawling**: Implementing deep crawling to follow links and extract data from multiple levels of a website posed a challenge. This was handled by setting a maximum depth for crawling and ensured that the crawler could efficiently manage the URLs it visits, avoiding infinite loops and redundant requests.

3. **API Rate Limits**: Another challenge was managing the rate limits imposed by the OpenAI API. his was addressed by implementing error handling and retries in the API call logic, ensuring that the application can gracefully recover from temporary issues.

4. **Instruction Clarity**: Ensuring that user instructions were clear and understandable by the model was crucial. The system and user prompts was designed to guide the model in rephrasing instructions for better comprehension, improving the quality of the filtered results.


## Conclusion

This project provides a robust framework for web data extraction and filtering using AI. By following the steps outlined above, you can easily set up and run the application to extract and filter content based on your specific needs.

For any issues or contributions, please feel free to open an issue or submit a pull request.