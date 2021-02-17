#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from pathlib import Path
import time
from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.trainset import Trainset
from surprise import accuracy



class pipeline(object):
    
    def load_ratings_from_file(self, ratings_filepath : Path) -> DatasetAutoFolds:
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratings = Dataset.load_from_file(ratings_filepath, reader)
        return ratings
    
    def load_ratings_from_surprise(self) -> DatasetAutoFolds:
        ratings = Dataset.load_builtin('ml-100k')
        return ratings

    def get_ratings(self, load_from_surprise : bool = True, ratings_filepath : Path = None) -> DatasetAutoFolds:
        if load_from_surprise:
            ratings = self.load_ratings_from_surprise()
        else:
            ratings = self.load_ratings_from_file(ratings_filepath)
        return ratings 
    
    def set_model_parameters(self, model_class: AlgoBase, **kwargs) -> AlgoBase:
        model = model_class(**kwargs)
        return model
    
    
    def train(self, param_model:AlgoBase, trainset:Trainset) -> AlgoBase:
        param_model.fit(trainset)
        return param_model

#     def train(self, model_class: AlgoBase, trainset: Trainset, **kwargs) -> AlgoBase:
#         model = model_class(**kwargs)
#         model.fit(trainset)
#         return model
    
    def model_prediction(self, model_class: AlgoBase, test: list) -> list:
        pred = model_class.test(test)
        return pred
    
    def evaluate_model_rmse_and_mae(self, param_pred:list) -> (np.float64, np.float64):
        """
        Parameters:
            param_pred: list
                        list of predictions
        
        Returns:
            numpy.float64, numpy.float64  
            returns rmse and mae accuracy
            
        """
        return accuracy.rmse(predictions=param_pred), accuracy.mae(predictions=param_pred)
    
    def evaluate_time_and_train(self, param_model:AlgoBase, trainset:Trainset) -> (float, AlgoBase):
        """
        Parameters:
            param_model: AlgoBase
                         model
                         
            trainset   : Trainset
                         training set to train model
        
        Returns:
            float, AlgoBase  
            returns time taken to train model (model_time) and trained model (model)
            
        """
        model_time = time.time()
        model = self.train(param_model, trainset)
        model_time = time.time() - model_time
        return model_time, model





