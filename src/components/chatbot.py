import openai
import pandas as pd
import json
import os
import PyPDF2
from llama_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from llama_index import ServiceContext
    
from langchain.chat_models import ChatOpenAI

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

filename_list = []
# # Define the directory path
directory = "Dataset"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file and its name starts with "Tags"
    if os.path.isfile(os.path.join(directory, filename)) and filename.startswith(
        "Job_Description"
    ):
        # Process the file
        filename_list.append(filename)

df_Description = pd.read_json(f"Dataset\{filename_list[0]}")
text = " ".join([i for i in df_Description["Job Description"]])
text = text.replace("\n", " ")

# Save text to txt file
f = open(r"textdata\all_text.txt", "w", encoding="utf-8")
f.write(text)
f.close()


def construct_index(directory_path: str) -> GPTSimpleVectorIndex:
    """
    Constructs a GPTSimpleVectorIndex from documents in the given directory_path.

    Args:
        directory_path: Path to the directory containing the documents.

    Returns:
        GPTSimpleVectorIndex object.
    """
    # Set maximum input size.
    max_input_size = 4096
    
    # Set number of output tokens.
    num_outputs = 500
    
    # Set maximum chunk overlap.
    max_chunk_overlap = 200
    
    # Set chunk size limit.
    chunk_size_limit = 600

    # Define LLM (ChatGPT gpt-3.5-turbo).
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )
    
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index = GPTSimpleVectorIndex.from_documents(documents,service_context=service_context)

    index.save_to_disk("index.json")

    return index


def ask_me_anything(question: str) -> str:
    """
    Given a question, return a response generated by the GPTSimpleVectorIndex.

    Args:
        question (str): The input question.

    Returns:
        str: The response generated by the GPTSimpleVectorIndex.
    """
    index = GPTSimpleVectorIndex.load_from_disk("index.json")
    response = index.query(question, response_mode="compact")

    return response.response
