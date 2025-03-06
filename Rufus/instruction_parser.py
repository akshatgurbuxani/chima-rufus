from openai import OpenAI
import json

class InstructionParser:
    """
    InstructionParser processes user instructions to filter content based on those instructions.

    This class utilizes the OpenAI gpt-3.5-turbo model to understand user instructions and filter the provided content.

    Attributes:
        api_key (str): The OpenAI API key for making requests to the gpt-3.5-turbo model.
    """

    def __init__(self, api_key, max_split_tokens=3000):
        """
        Initializes the InstructionParser with the provided OpenAI API key.

        :param api_key: The OpenAI API key for interacting with the gpt-3.5-turbo model.
        :param max_split_tokens: The maximum number of tokens per API call.
        """
        self.client = OpenAI(api_key=api_key)
        self.max_split_tokens = max_split_tokens

    def filter_content(self, content, instructions):
        """
        Filters the provided content based on user instructions using the gpt-3.5-turbo model.

        :param content: The raw content in JSON format to be filtered.
        :param instructions: User instructions for filtering the content.
        :return: A dictionary containing the original content and the filtered content in JSON format.
        """
        # Split the content into manageable chunks
        content_chunks = self._split_content(content)

        filtered_results = []

        for chunk in content_chunks:
            prompt = (
                "You are a helpful assistant that filters content based on user instructions.\n\n"
                "Your task is to analyze the provided content, focusing on the 'headings' and 'text' fields in JSON format, "
                "and return only the relevant content based on user instructions.\n\n"
                f"User instructions: {instructions}\n\n"
                "The following is a structured JSON document:\n"
                f"{json.dumps(chunk, indent=2)}\n\n"
                "Filter the content accordingly and return **only the filtered content** in valid JSON format."
            )

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that filters content based on user instructions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.0
                )

                # Log the raw response for debugging
                print("Raw response from LLM:", response)

                # Extract response and ensure JSON validity
                filtered_content = response.choices[0].message.content.strip()

                # Clean up the filtered content by removing the code block formatting
                if filtered_content.startswith("```json") and filtered_content.endswith("```"):
                    filtered_content = filtered_content[7:-3].strip()  # Remove the ```json and the closing ```

                try:
                    filtered_results.append(json.loads(filtered_content))  # Ensure proper JSON parsing
                except json.JSONDecodeError:
                    print(f"Error parsing LLM response: {filtered_content}")

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
            if len(current_chunk) >= self.max_split_tokens:
                chunks.append(current_chunk)
                current_chunk = []  # Reset for next chunk

        # Handle last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks