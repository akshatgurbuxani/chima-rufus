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