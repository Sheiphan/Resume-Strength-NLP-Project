import pandas as pd
import os
import re


filename_list = []
# # Define the directory path
directory = "D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file and its name starts with "Tags"
    if os.path.isfile(os.path.join(directory, filename)) and filename.startswith("Tags"):
        # Process the file
        filename_list.append(filename)
# print(filename_list)

Skills = set()
for k in range(len(filename_list)):
    df_Skills = pd.read_json(f'D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset\{filename_list[k]}')
    for i in range(len(df_Skills['Tags'])):
        for j in range(len(df_Skills['Tags'][i])):
            Skills.add(df_Skills['Tags'][i][j])
            continue
Skills_list = list(Skills)
# print(Skills_list)

data_skills = {'Entity': ['SKILLS'] * len(Skills), 'Skills': Skills_list}

Skills_df = pd.DataFrame(data_skills)
print(Skills_df)

# Skills_df.to_csv('All_Skills.csv', index=False, header=True)


# ----------------------------------------------------------------


filename_list_Location = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file and its name starts with "Tags"
    if os.path.isfile(os.path.join(directory, filename)) and filename.startswith("Job_descriptions"):
        # Process the file
        filename_list_Location.append(filename)
        
Location = set()
for k in range(len(filename_list_Location)):
    df_Location = pd.read_json(f'D:\Python\Resume_NLP_Project\Resume-Strength-NLP-Project\Dataset\{filename_list_Location[k]}')
    for i in range(len(df_Location['Location'])):
        Location.add(df_Location['Location'][i])
        continue
Location_list = list(Location)
# print(Location_list)

Location_list_split = set()
# Iterate over each string in the list and split it into words
for word in Location_list:
    # Split the string into words using the space character as the separator
    # word = word.replace(',','')
    word = re.sub(r'[^a-zA-Z\s]+', '', word)
    word = ' '.join(re.findall(r'\b[A-Z][a-zA-Z]*\b', word))
    words = word.split()
    # Print the words
    for w in words:
        Location_list_split.add(w)
        Location_list_split_list = list(Location_list_split)
Location_list_split_list = [char for char in Location_list_split_list if char != '-']


data_Location = {'Entity': ['LOC'] * len(Location_list_split_list), 'Skills': Location_list_split_list}

Location_df = pd.DataFrame(data_Location)
print(Location_df)

# Location_df.to_csv('All_Location.csv', index=False, header=True)


# ----------------------------------------------------------------

Complete_df = pd.concat([Skills_df, Location_df], ignore_index=True, axis=0)
Complete_df.to_csv('All_Complete.csv', index=False, header=True)
