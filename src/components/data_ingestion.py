import os
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTraining

from dataclasses import dataclass

@dataclass
class dataIngestionConfig():
    dataPath = os.path.join(os.getcwd(),"ARTIFACTS","DATA","df.csv")

class dataIngestion():
    def __init__(self):
        self.dataIngestionConfig = dataIngestionConfig()

    def initiateDataIngestion(self):
        try:
            logger.info("Starting Data Ingestion")

            df = pd.read_csv("/Users/sanketsaxena/Desktop/recommendationSystem/ARTIFACTS/DATA/df.csv")

            logger.info("Data Ingestion completed")

            return (
                df,
                self.dataIngestionConfig.dataPath
                )

        except Exception as e:
            raise CustomException("Something wrong in initiateDataIngestion method of dataIngestion class")


if __name__ == "__main__":
    df = pd.read_csv("/Users/sanketsaxena/Desktop/recommendationSystem/ARTIFACTS/DATA/transformed.csv")
    data_transformation = dataTransformation()
    df_transformed, df_trans_path = data_transformation.initiateDataTransformation(df=df)
    model_training = modelTraining()
    w2vmodelPath,pcaModelPath,pcaDataTransformed,pca_data = model_training.initiateModelTraining(df_transformed)
    print(w2vmodelPath, pcaModelPath, pcaDataTransformed)
    print(type(pca_data))