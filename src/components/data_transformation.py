import os
import re
import ast 

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import shortDesc

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

from dataclasses import dataclass

@dataclass
class dataTransformationConfig():
    dfTransformedPath : str = os.path.join(os.getcwd(),"ARTIFACTS","df_Transformed.csv")

class dataTransformation():
    def __init__(self):
        self.dataTransformationConfig = dataTransformationConfig()

    def initiateDataTransformation(self,df: pd.DataFrame):
        try:
            """logger.info("Text Cleaning and Preprocessing Started")
            logger.info("Removing all the NaN values from the videos and storyline column")

            df['videos'] = df['videos'].fillna('[]')
            df['storyline'] = df['storyline'].fillna('')

            logger.info("Removed all NaN Values from title column")
            logger.info("Converting videos and genres back to original data structure")
            
            df['videos'] = df['videos'].apply(ast.literal_eval)
            df['genres'] = df['genres'].apply(ast.literal_eval)
            df['genres'] = df['genres'].apply(lambda x: ' '.join(x))

            logger.info("Converted videos and genres back to original data structure")
            logger.info("Concatenating two columns to form Single Description Column")

            df['description'] = df['genres']+" "+ df['storyline']

            logger.info("Concatenated genres and storyline to form description olumn")
            logger.info("Shortening the description")

            df['description'] = df['description'].apply(shortDesc)

            logger.info("Shortened the description")
            logger.info("Lowering the case of description column")

            df['description'] = df['description'].str.lower()

            logger.info("Lowered the case of titleDesc column")
            logger.info("Removing Punctuations from the titleDesc column")

            df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

            logger.info("Removed Punctuations from the titleDesc column")
            logger.info("Removing stopwords from the titleDescr column")

            df['description'] = df['description'].apply(lambda x: ' '.join([i for i in x.split(" ") if i not in stopwords.words('english')]))

            logger.info(f"Removed stopwords from the titleDescr column, Example value looks like : {df['titleDesc'][0]}")"""
            logger.info("Spliting the titleDescr column into list and removing duplicate words")

            df['description'] = df['description'].apply(lambda x: list(dict.fromkeys(x.split(" "))))

            logger.info("Splited the titleDescr column into list and removed duplicate words")
            logger.info("Text Cleaning and Preprocessing completed")
            logger.info("Saving the transformed df into Artifact Folder")

            df.to_csv(self.dataTransformationConfig.dfTransformedPath)
            
            logger.info("Saved the transformed df into Artifact Folder")

            return (
                df,
                self.dataTransformationConfig.dfTransformedPath
            )
        except Exception as e:
            raise CustomException("Something wrong in initiateDataTransformation method of dataTransformation class")

