from src.utils.logging import logger
from src.utils.exceptions import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import mlflow
import os

@dataclass
class PredictionPipelineConfig:
  modelUri: str = "runs:/9830f522484543dabf0f4ad6e699f92d/CatBoostModel"
  featureEncoderObject: str = os.path.join("artifacts", "featureEncoderObject.joblib")
  kmeansModel: str = os.path.join("artifacts", "kmeansModel.joblib")

class PredictionPipeline:
  def __init__(self):
    self.predictionPipelineConfig = PredictionPipelineConfig()
  
  def predictResult(self, array: np.ndarray) -> int:
    # reshaping the array
    array = pd.DataFrame(array, index = ["job", "month", "balance", "duration", "campaign", "poutcome", "contacted_previously", "loans"]).T

    # encode features
    featureEncoder = joblib.load(self.predictionPipelineConfig.featureEncoderObject)
    array = featureEncoder.transform(array)
    print(array)

    # perform clustering
    kmeansModel = joblib.load(self.predictionPipelineConfig.kmeansModel)
    array = np.append(array, kmeansModel.predict(array).reshape((1, -1)))

    # loading the model
    modelName = "CatBoostModel"
    modelVersion = 28
    model = mlflow.pyfunc.load_model(self.predictionPipelineConfig.modelUri)
    result = model.predict(array)
    result = int(result)

    return result