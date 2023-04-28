import pandas as pd
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.models import save_model
from keras.callbacks import EarlyStopping



import matplotlib.pyplot as plt
# !pip install mplcursors
# import mplcursors



# code to create just headings of df from skills of different job roles

# skills = set()

# for filename in os.listdir('dataset_1'):
#     prev_skills_lenght = len(skills)
#     with open(os.path.join('dataset_1', filename)) as f:
#         data = json.load(f)
#         for skill_set in data['Tags'].values():
#             skills.update(skill_set)
    
#     print(len(skills)-prev_skills_lenght,"skills extracted from:",filename)
# print(len(skills),"total skills extracted")
# skills_df = pd.DataFrame(columns=list(skills))

# # code to OHE skills
# for filename in os.listdir('dataset_1'):
#     with open(os.path.join('dataset_1', filename)) as f:
#         data = json.load(f)
#         for i, skill_set in data['Tags'].items():
#             row = {skill: int(skill in skill_set) for skill in skills}
#             row['job_role'] = filename[:-5]
#             skills_df = skills_df.append(row, ignore_index=True)
            
# skills_df.to_csv('jobs&skills.csv', index=False)

# skills_df = pd.read_csv("jobs&skills.csv")

# roles = skills_df['job_role'].unique()
class DataManipulation:
    def job_data(roles, skills_df):
        
        all_job_data = dict()

        for role in roles:
            temp_df = skills_df[skills_df['job_role'] == role]
            temp_df = temp_df.iloc[:,:-1]
            lst = temp_df.sum(axis=0)
            percentwise_counts = 100 * lst / len(temp_df)
            percentwise_counts = percentwise_counts.sort_values()
            #  skills which are not present in any job posting for the role
                    
            print("\n\n",role)
            
            most_important_skill = percentwise_counts.idxmax()
            percent_of_MIS = percentwise_counts.max()
            print("   most important skill: ",most_important_skill,"that is",percent_of_MIS,"%")
            
            skills_ab = [index for index, value in percentwise_counts.items() if value >= 0 and value <= 0.000000001]
            print("   absent skills: ", len(skills_ab), skills_ab)

            #Top skills required for each role
            skills_most = [index for index, value in percentwise_counts.items() if value >= 40 and value <= 100]
            print("   most common skills: ",len(skills_most), skills_most)

            #Most rare skills for each role
            skills_least = [index for index, value in percentwise_counts.items() if value >= 0.0000001 and value <= 3]
            print("   most rare and useless skills: ",len(skills_least),skills_least)
            
            one_data = [most_important_skill, percent_of_MIS,
                    len(skills_most), skills_most,
                    ]
            
            all_job_data.update({f'{role}': 
                                            {'most_important_skill' : most_important_skill,
                                             'percent_of_MIS': percent_of_MIS,
                                             'num_skills_most' : len(skills_most),
                                             'skills_most' : skills_most,
                                             } 
                                 }
                                )
            
        return all_job_data


    # creating df who has dropped skills which are for less than 3% of job postings of that role
    def jobs_skills_final_csv(roles,skills_df):
        jobs_skills_final = pd.DataFrame(columns=skills_df.columns)

        for role in roles:
            temp_df = skills_df[skills_df['job_role'] == role]
        
            lst = temp_df.iloc[:,:-1].sum(axis=0)
            percentwise_counts = 100 * lst / len(temp_df)
        
            # Most rare skills for each role
            skills_least = [index for index, value in percentwise_counts.items() if value >= 0.0000001 and value <= 3]
        
            # Set least common skills to 0
            temp_df[skills_least] = 0
            
            # Append modified DataFrame to jobs_skills_final
            jobs_skills_final = jobs_skills_final.append(temp_df)

        # Reset index of final DataFrame
        jobs_skills_final.reset_index(drop=True, inplace=True)
        
        jobs_skills_final.to_csv('jobs_skills_final.csv', index=False)
        


# job_data(roles, skills_df)

def neural_network(jobs_skills_final):
    le = LabelEncoder()
    jobs_skills_final['job_role'] = le.fit_transform(jobs_skills_final['job_role'])

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from sklearn.model_selection import train_test_split

    # Prepare the data
    X = jobs_skills_final.iloc[:, :-1].values
    y = jobs_skills_final.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model architecture
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Save the model
    save_model(model, 'model.h5')

# jobs_skills_final = pd.read_csv('jobs_skills_final.csv')

# neural_network(jobs_skills_final)