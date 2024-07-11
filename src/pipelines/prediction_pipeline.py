import os
import pickle
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

from src.utils import calculate_cosine_similarity
from src.exception import CustomException
from src.logger import logger

class predictionPipeline():
    def __init__(self):
        pass

    def predict(self, X: str):
        try:
            logger.info("Doing text preprocessing")

            X = X.lower()
            X =re.sub(r'[^\w\s]', '', X)
            X = list(dict.fromkeys([i for i in X.split(" ") if i not in stopwords.words('english')]))
            X = pd.DataFrame({"description" : [X]})

            logger.info(f"Text preprocessing done. After text preprocessing X is : {X}")
            logger.info("Loading the w2vModel")

            w2vmodel = pickle.load(open("/app/ARTIFACTS/w2vmodel.pkl",'rb'))

            logger.info("Loaded w2vModel")
            logger.info("Applying w2vModel on input X")

            X['description'] = X['description'].apply(lambda x: [w2vmodel.wv[i] for i in x if i in w2vmodel.wv])
            X['description'] = X['description'].apply(lambda x: np.mean(x, axis=0))

            logger.info(f"Applied w2vModel on input X. X = {X.head(0)}")
            logger.info("Loading the pcaModel")

            pcamodel = pickle.load(open("/app/ARTIFACTS/pcamodel.pkl",'rb'))

            logger.info("Loaded pcamodel")
            logger.info("Applying pcamodel on input X")

            X_array = np.vstack(X['description'].apply(np.array))
            X_pca = pcamodel.transform(X_array)

            logger.info(f"Applied pcamodel on input X. X = {X_pca}")
            logger.info("Finding top 10 similar products by calculating cosine similarity")

            pc_data = np.load("/app/ARTIFACTS/pcaTransformed.npy")
            x_trans = pd.read_csv("/app/ARTIFACTS/df_Transformed.csv")

            top_10_similar_items = calculate_cosine_similarity(pca_data=pc_data,X_pca=X_pca,X_transformed=x_trans)

            logger.info(f"Found top 10 similar products and they are {top_10_similar_items}")
            return (
                top_10_similar_items
            )


        except:
            raise CustomException("Something wrong in predict method of predictionPipeline class")
        

if __name__ == "__main__":
    predict_pipeline = predictionPipeline()
    predict_pipeline.predict("Olive Satchel")
