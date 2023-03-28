import openai
import pandas as pd
import json
import os
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.llms import OpenAI
from IPython.display import Markdown, display

filename_list = []
# # Define the directory path
directory = "D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file and its name starts with "Tags"
    if os.path.isfile(os.path.join(directory, filename)) and filename.startswith("Job_Description"):
        # Process the file
        filename_list.append(filename)

df_Description = pd.read_json(f'D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset\{filename_list[0]}')
text = ' '.join([i for i in df_Description['Job Description']])
text = text.replace('\n',' ')

# Save text to txt file
f = open(r"D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\textdata\all_text.txt", "w", encoding="utf-8") 
f.write(text)
f.close()

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    
    chunk_size_limit = 600


    # define LLM (ChatGPT gpt-3.5-turbo)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index


def ask_me_anything(question):

    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(question, response_mode="compact")

    # display(Markdown(f"You asked: <b>{question}</b>"))
    # display(Markdown(f"Bot says: <b>{response.response}</b>"))
    print(f"You asked: {question}")
    print(f"Bot says: {response.response}")
    
# Set OpenAI key
os.environ["OPENAI_API_KEY"] = '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'

construct_index(r'D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\textdata')

question = input('Enter a question (Type Exit to Exit from the chatbot): ')

while question!='Exit':
    ask_me_anything(question)
    question = input('Enter a question (Type Exit to Exit from the chatbot): ')
