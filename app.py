from flask import Flask, request, render_template
import numpy as np
# import re
from src.components.NER import extract_pdf, ner ,KeyphraseExtractionPipeline
from src.components.chatbot import ask_me_anything, construct_index
from pdfminer.high_level import extract_pages, extract_text
from transformers import AutoTokenizer, AutoModelForTokenClassification

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
    print("Get")
    if request.method == "GET":
        return render_template("upload.html")
    else:
        print("Post")
        pdf_file = request.files["pdf_file"]
        if pdf_file:
            print('load pdf')
            pdf_file_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
            print('save pdf')
            pdf_file.save(pdf_file_path)
            print('extract')
            pdf_text = extract_pdf(pdf_file_path)
            print(pdf_text)
            # print(pdf_file_path)
            # text = extract_text(str(pdf_file_path))
            # text_replace = text.replace('\n',' ')
            # return text_replace
            skills = []
            
            print('build model')

            # Load pipeline
            model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
            extractor = KeyphraseExtractionPipeline(model=model_name)
            
            keyphrases_1 = extractor(pdf_text)
            print(type(keyphrases_1))
            
            
            tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
            model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
            print('keyprae')
            
            keyphrases_2 = ner(pdf_text, model, tokenizer)
            
            for i in range(len(keyphrases_2)):
                # print(keyphrases[i])
                if keyphrases_2[i]['entity_group'] == 'MISC':
                    skills.append(keyphrases_2[i]['word'])
                    print(keyphrases_2[i]['word'])
                    
            skills = np.array(skills)
            keyphrases = np.concatenate((keyphrases_1,skills))
            print(keyphrases)
            keyphrases = list(keyphrases)
            return render_template("upload.html", keyphrases=keyphrases)

            # return 'PDF file uploaded and saved successfully!'
        else:
            return "No PDF file uploaded"


@app.route("/answer", methods=["GET", "POST"])
def answer():
    if request.method == "GET":
        return render_template("chatbot.html")

    else:
        # construct_index(r"textdata")

        question = request.form["question"]

        while question != "Exit":
            answer = ask_me_anything(question)
            return render_template("chatbot.html", answer=answer)
            question = input("Enter a question (Type Exit to Exit from the chatbot): ")
            
        # Do something to get the answer based on the question
        # answer = "The answer to your question is..."


if __name__ == "__main__":
    app.run()
