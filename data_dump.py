import pymongo
import pandas as pd
import json
from pymongo import MongoClient, InsertOne

client = pymongo.MongoClient("mongodb+srv://Sheiphan_MongoDB:EBEnezeR123@cluster0.iewrgfp.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE = (r"D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset\Job_descriptions_Analyst.csv")
DATABASE_NAME = "ANALYST"
COLLECTION_NAME = "Job_descriptions_Analyst"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    print(f"Number of Elements: {df.shape}")
    
    df.reset_index(drop = True, inplace=True)
    
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record)
              
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)