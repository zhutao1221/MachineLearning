# -*- coding: utf-8 -*-
import json
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

def linearModel(request):
    linreg = joblib.load("MachineLearningService/model_save/train_model.m")
    X = [[0, 0], [1, 1]]
    y = linreg.predict(X)
    y1 = y.tolist()
    python2json = {}
    python2json["predicted"] = y1
    return HttpResponse(json.dumps(python2json), content_type="application/json")