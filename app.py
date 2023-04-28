from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_pages, extract_text
from transformers import AutoTokenizer, AutoModelForTokenClassification
# import re

from src.components.NER import extract_pdf, ner ,KeyphraseExtractionPipeline
from src.components.chatbot import ask_me_anything, construct_index
from src.components.resume import DataManipulation, Neural_Net
# from src.components.resume import neural_network

from src.exception import CustomException
from src.logger import logging

import os

app = Flask(__name__)

UPLOAD_FOLDER = "textdata"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return render_template("index.html")


# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['resume']
#     file.save('uploaded_file.pdf')
#     return 'File uploaded successfully'


@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template("upload.html")
    else:
        pdf_file = request.files["pdf_file"]
        
        if pdf_file:

            pdf_file_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)

            pdf_file.save(pdf_file_path)

            pdf_text = extract_pdf(pdf_file_path)

            skills = []
            
            # Load pipeline
            model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
            extractor = KeyphraseExtractionPipeline(model=model_name)
            
            keyphrases_1 = extractor(pdf_text)
            print(type(keyphrases_1))
            
            
            tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
            model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
            
            keyphrases_2 = ner(pdf_text, model, tokenizer)
            
            for i in range(len(keyphrases_2)):
                # print(keyphrases[i])
                if keyphrases_2[i]['entity_group'] == 'MISC':
                    skills.append(keyphrases_2[i]['word'])
                    # print(keyphrases_2[i]['word'])
                    
            skills = np.array(skills)
            keyphrases = np.concatenate((keyphrases_1,skills))
            print(keyphrases)
            keyphrases = list(keyphrases)
            
            logging.info('Keyphrases Token been generated')
            
            skills_df = pd.read_csv("jobs&skills.csv")
            roles = skills_df['job_role'].unique()
            
            job = DataManipulation(roles, skills_df)
            all_data  = job.job_data()
            print("ALL DATA")
            # print(all_data)
            job.jobs_skills_final_csv()
            
            logging.info('Job Statistics Related to Roles Generated')
            
            jobs_skills_final = pd.read_csv('jobs_skills_final.csv')
            Neural_Object = Neural_Net(jobs_skills_final, roles, skills_df)
            Neural_Object.train()
            
            return render_template("upload.html", keyphrases=keyphrases, all_data=all_data)

            
            # return 'PDF file uploaded and saved successfully!'
        else:
            return "No PDF file uploaded"


@app.route("/answer", methods=["GET", "POST"])
def answer():
    if request.method == "GET":
        return render_template("chatbot.html")

    else:
        if 'index.json' in os.listdir():
            logging.info('index.json file exists in the directory')
        else:
            logging.info('index.json file does not exist in the directory')
            construct_index(r"textdata")

        question = request.form["question"]

        while question != "Exit":
            answer = ask_me_anything(question)
            logging.info('Got answer')
            return render_template("chatbot.html", answer=answer)
            question = input("Enter a question (Type Exit to Exit from the chatbot): ")
        
        # Do something to get the answer based on the question
        # answer = "The answer to your question is..."


if __name__ == "__main__":
    app.run()
