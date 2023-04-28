import pandas as pd
import json
import os
from typing import List
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from src.exception import CustomException
from src.logger import logging




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
    """
    This class performs data manipulation on job roles and skills.
    """

    def __init__(self, roles: list, skills_df: pd.DataFrame):
        """
        Constructor for the DataManipulation class.

        Parameters:
        roles (list): List of job roles.
        skills_df (pd.DataFrame): Dataframe of job roles and their respective skills.
        """
        self.roles = roles
        self.skills_df = skills_df
        self.job_data_dict = {}
        self.jobs_skills_final_df = pd.DataFrame(columns=skills_df.columns)

    def job_data(self) -> dict:
        """
        This method calculates the most important skill, percentage of most important skill, 
        skills that are most important and least important for each job role.

        Returns:
        dict: Dictionary containing job roles, their most important skill and related information.
        """
        for role in self.roles:
            temp_df = self.skills_df[self.skills_df['job_role'] == role]
            temp_df = temp_df.iloc[:,:-1]
            lst = temp_df.sum(axis=0)
            percentwise_counts = 100 * lst / len(temp_df)
            percentwise_counts = percentwise_counts.sort_values()
            
            most_important_skill = percentwise_counts.idxmax()
            percent_of_MIS = percentwise_counts.max()
            
            skills_ab = [index for index, value in percentwise_counts.items() if value >= 0 and value <= 0.000000001]
            
            skills_most = [index for index, value in percentwise_counts.items() if value >= 40 and value <= 100]
            
            skills_least = [index for index, value in percentwise_counts.items() if value >= 0.0000001 and value <= 3]
            
            one_data = {'most_important_skill': most_important_skill, 
                        'percent_of_MIS': percent_of_MIS,
                        'num_skills_most': len(skills_most), 
                        'skills_most': skills_most}
            
            self.job_data_dict[role] = one_data
            
            
        logging.info('Dictionary containing job roles, their most important skill and related information.')

        return self.job_data_dict
    
    def jobs_skills_final_csv(self) -> None:
        """
        This method creates a csv file of the final dataframe containing job roles and their respective skills.
        """
        for role in self.roles:
            temp_df = self.skills_df[self.skills_df['job_role'] == role]
        
            lst = temp_df.iloc[:,:-1].sum(axis=0)
            percentwise_counts = 100 * lst / len(temp_df)
        
            skills_least = [index for index, value in percentwise_counts.items() if value >= 0.0000001 and value <= 3]
        
            temp_df[skills_least] = 0
            
            self.jobs_skills_final_df = self.jobs_skills_final_df.append(temp_df)

        self.jobs_skills_final_df.reset_index(drop=True, inplace=True)
        
        self.jobs_skills_final_df.to_csv('jobs_skills_final.csv', index=False)
        
        logging.info('Creates a csv file of the final dataframe containing job roles and their respective skills.')




# job_data(roles, skills_df)


class Neural_Net(DataManipulation):
    """
    Neural network class to prepare data and train a model for job classification.
    """
    def __init__(self, jobs_skills_final: pd.DataFrame, roles: List[str], skills_df: pd.DataFrame):
        """
        Initialize a Neural_Net object with jobs_skills_final, roles and skills_df dataframes.

        Parameters:
        -----------
        jobs_skills_final : pandas dataframe
            A dataframe containing job roles and skills.
        roles : list of str
            A list of job roles.
        skills_df : pandas dataframe
            A dataframe containing skills and their corresponding level of expertise.
        """
        super().__init__(roles, skills_df)
        self.jobs_skills_final = jobs_skills_final
        self.le = LabelEncoder()
        
    def _prepare_data(self):
        """
        Prepare the data for training and testing the model.
        """
        self.jobs_skills_final['job_role'] = self.le.fit_transform(self.jobs_skills_final['job_role'])
        X = self.jobs_skills_final.iloc[:, :-1].values
        y = self.jobs_skills_final.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    def _define_model(self):
        """
        Define the neural network model architecture.
        """
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=self.X_train.shape[1], activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(np.unique(self.y_train)), activation='softmax'))
        
    def _compile_model(self):
        """
        Compile the neural network model.
        """
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def _define_callbacks(self):
        """
        Define the callbacks for the neural network model.
        """
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
        
    def train(self):
        """
        Train the neural network model and save it to disk.
        """
        self._prepare_data()
        self._define_model()
        self._compile_model()
        self._define_callbacks()
        history = self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=32, validation_data=(self.X_test, self.y_test), callbacks=self.callbacks)
        save_model(self.model, 'model.h5')
        
        logging.info('Train the neural network model and save it to disk.')