import os
import time
import mlflow
import gensim

import dagshub
dagshub.init(repo_owner='sanket-profile', repo_name='recommendationSystem', mlflow=True)

import numpy as np
import multiprocessing

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from mlflow.sklearn import log_model

from dataclasses import dataclass

@dataclass
class modelTrainingConfig():
    w2vmodelPath : str = os.path.join(os.getcwd(),"ARTIFACTS","w2vmodel.pkl")
    pcaModelPath : str = os.path.join(os.getcwd(),"ARTIFACTS","pcamodel.pkl")
    pcaDataTransformed : str = os.path.join(os.getcwd(),"ARTIFACTS","pcaTransformed.npy")

class modelTraining():
    def __init__(self):
        self.modelTrainingConfig = modelTrainingConfig()

    def initiateModelTraining(self,df):
        try:
            mlflow.set_tracking_uri(
                    "https://dagshub.com/sanket-profile/recommendationSystem.mlflow"
                    )
            mlflow.set_experiment("Testing Recommendations")
            mlflow.autolog()
            with mlflow.start_run():
                logger.info("Starting the training process")
                logger.info("Initializing W2V model with 100 feature output")

                w2v_model = Word2Vec(
                    min_count = 1,
                    window = 7,
                    sg = 0,
                    alpha = 0.03,
                    min_alpha = 0.0007,
                    negative = 20,
                    workers = multiprocessing.cpu_count()
                )

                logger.info("Initialized W2V model")
                logger.info("Building Vocab Count")

                t = time.time()
                w2v_model.build_vocab(df['description'], progress_per=10000)

                logger.info("Completed Building Vocab Count")     
                logger.info("Starting training W2Vmodel")

                w2v_model.train(df['description'], total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
                log_model(w2v_model,"w2vmodel")

                logger.info("Completed Training W2Vmodel")
                logger.info("Converting each word in titleDescr column(LIST) to feature representation")      

                df['description'] = df['description'].apply(lambda x: [w2v_model.wv[i] for i in x])

                logger.info("Converted each word in titleDescr column(LIST) to feature representation") 
                logger.info("Applying mean to titleDescr Colum to form a single sentence encoding")

                df['description'] = df['description'].apply(lambda x: np.mean(x, axis=0))

                logger.info("Applied mean and formed the sentence embeddings")
                logger.info("Starting training of PCA model to conver 100 dimensions to 50 dimensions")

                desc_pre1_array = np.vstack(df['description'].apply(np.array))
                pca = PCA(100)
                pca.fit(desc_pre1_array)
                pca_data = pca.transform(desc_pre1_array)

                logger.info("Trained the PCA model and saved the dimensionality reduced data")
                logger.info("Saving W2Vmodel and PCA model into pickle file in Artifact Folder")

                save_object(self.modelTrainingConfig.w2vmodelPath,w2v_model)
                save_object(self.modelTrainingConfig.pcaModelPath,pca)

                logger.info("Saved W2Vmodel and PCA model into pickle file")
                logger.info("Saving PCA data into Artifacts Folder")

                np.save(self.modelTrainingConfig.pcaDataTransformed,pca_data)

                logger.info("Saved PCA data into Artifacts Folder")

                return (
                    self.modelTrainingConfig.w2vmodelPath,
                    self.modelTrainingConfig.pcaModelPath,
                    self.modelTrainingConfig.pcaDataTransformed,
                    pca_data
                )
            
        except Exception as e:
            raise CustomException("Something wrong in initiateModelTraining method of modelTraining class")