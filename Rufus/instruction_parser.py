import openai
import json

class InstructionParser:
    """
    InstructionParser processes user instructions to filter content based on those instructions.

    This class utilizes the OpenAI GPT-4o model to understand user instructions and filter the provided content.

    Attributes:
        api_key (str): The OpenAI API key for making requests to the GPT-4o model.
    """

    def __init__(self, api_key, max_split_tokens=3000):
        """
        Initializes the InstructionParser with the provided OpenAI API key.

        :param api_key: The OpenAI API key for interacting with the GPT-4o model.
        :param max_split_tokens: The maximum number of tokens per API call.
        """
        openai.api_key = api_key
        self.max_split_tokens = max_split_tokens

    def filter_content(self, content, instructions):
        """
        Filters the provided content based on user instructions using the GPT-4o model.

        :param content: The raw content in JSON format to be filtered.
        :param instructions: User instructions for filtering the content.
        :return: A dictionary containing the original content and the filtered content in JSON format.
        """
        # Split the content into manageable chunks
        content_chunks = self._split_content(content)

        filtered_results = []

        for chunk in content_chunks:
            # Prepare the prompt for the LLM
            prompt = (
                "You are a helpful assistant that filters content based on user instructions. "
                "Your task is to understand the user's request and filter the provided content accordingly by looking at the 'headings' and 'texts' columns from JSON.\n\n"
                f"The following is a set of content in JSON format:\n"
                f"{json.dumps(chunk, indent=2)}\n\n"
                f"User instructions: {instructions}\n\n"
                "Please rewrite the instructions if necessary for clarity, and then filter the content based on these instructions by looking at the 'headings' and 'texts' columns from JSON."
                "Make suree to return the filtered content in JSON format."
            )

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that filters content based on user instructions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,  # Adjust as needed based on expected output size
                    temperature=0.0
                )

                # Extract the filtered content from the response
                filtered_content = response['choices'][0]['message']['content']
                filtered_results.append(json.loads(filtered_content))  # Convert the filtered content back to JSON

            except Exception as e:
                print(f"Error filtering content for chunk: {e}")


        return filtered_results 

    def _split_content(self, content):
        """
        Splits the content into smaller chunks based on the maximum token limit.

        :param content: The raw content in JSON format to be split.
        :return: A list of content chunks.
        """
        # Convert content to a string and estimate token count
        content_str = json.dumps(content)
        tokens = content_str.split()  # Simple tokenization by whitespace
        chunks = []
        current_chunk = []

        for token in tokens:
            current_chunk.append(token)
            if len(' '.join(current_chunk)) >= self.max_split_tokens:
                chunks.append(json.loads(' '.join(current_chunk)))  # Convert back to JSON
                current_chunk = []

        if current_chunk:
            chunks.append(json.loads(' '.join(current_chunk)))  # Add the last chunk

        return chunks