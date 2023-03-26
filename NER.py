import pandas as pd
from simpletransformers.ner import NERModel,NERArgs

data = pd.read_csv("D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset\All_Skills.csv" )
# print(data.head(10))


X= data["Skills"]
Y =data["Entity"]

label = data["labels"].unique().tolist()
print(label)