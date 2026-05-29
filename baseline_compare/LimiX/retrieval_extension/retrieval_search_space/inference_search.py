import optuna
from typing import Literal
import os
import json

from sklearn.metrics import f1_score, precision_score, mean_squared_error, r2_score, accuracy_score

from inference.predictor import LimiXPredictor
from retrieval_extension.retrieval_search_space.init_search_space import generate_search_space
import numpy as np
import torch

from utils.inference_utils import auc_metric


class RetrievalSearchHyperparameters:
    def __init__(self, args, trainX, trainy, testX, testy, attention_score=None):
        self.args = args
        self.study = optuna.create_study(direction="maximize")
        self.trainX = trainX
        self.trainy = trainy
        self.testX = testX
        self.testy = testy
        self.attention_score = attention_score

    def search(self, method=None, metric: Literal["AUC", "accuracy", "f1"] = "AUC", n_trials: int = 1000, inference_config=None,task_type="cls"):
        if isinstance(inference_config, str):
            if os.path.isfile(inference_config):
                with open(inference_config, 'r') as f:
                    inference_config = json.load(f)
            else:
                raise ValueError(f"inference_config is not a config file path: {inference_config}")
        self.study.optimize(lambda trial: self.optuna_inference(trial, method, metric, inference_config,task_type), n_trials=n_trials)
        best_params = self.study.best_params
        print(f"best_params: {best_params}")
        print(f"best metric on vaildation: {self.study.best_value}")
        return best_params, self.study.best_value

    def optuna_inference(self, trial, method, metric: Literal["AUC", "accuracy", "f1"] = "accuracy",
                          inference_config=None,task_type="cls"):
        param = generate_search_space(trial, self.args)
        print(f"current params: {param}")
        for i in range(len(inference_config)):
            inference_config[i]["retrieval_config"].update(param)

        classifier = LimiXPredictor(device=torch.device(f"cuda:{self.args['device_id']}" if torch.cuda.is_available() else "cpu"),
                                    model_path=self.args["model_path"],
                                    inference_config=inference_config)
        if task_type == "cls":
            prediction_ = classifier.predict(self.trainX, self.trainy, self.testX, task_type="Classification")
            if metric == "AUC":
                return float(auc_metric(self.testy, prediction_))
            elif metric == "accuracy":
                return float(accuracy_score(self.testy, np.argmax(prediction_, axis=1)))
            elif metric == "f1":
                return float(f1_score(self.testy, np.argmax(prediction_, axis=1), average='macro'))
        else:
            y_mean = self.trainy.mean()
            y_std = self.trainy.std()
            y_train_normalized = (self.trainy - y_mean) / y_std
            y_test_normalized = (self.testy - y_mean) / y_std
            y_pred = classifier.predict(self.trainX, y_train_normalized, self.testX, task_type="Regression")
            y_pred = y_pred.to('cpu')
            r2 = r2_score(y_test_normalized, y_pred)
            return float(r2)
