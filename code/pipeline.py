#########################################
### DBgriff and Robikscube Kaggle Project
#########

### Import Packages

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from time import time
import logging
from sklearn.model_selection import train_test_split
# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing

#SK-learn libraries for transformation and pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer

# Custom classes for this project
import feature_engineering as fe

##############################################
# LOAD THE DATASETS
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

##############################################
# Define pipeline


#### Example below #######################


questions = ('question1', 'question2', )

pipeline = Pipeline([   
    ('questions', Pipeline([
        ('date', fe.FuzzyFeatures()),
        ('drop_datetime', fe.WordMatchShare()),
        ('SentimentFeatures', fe.SentimentFeatures()),
        ('LengthShare', fe.SentimentFeatures()),
        ('scale', StandardScaler()),    
    ])),
('to_dense', preprocessing.FunctionTransformer(lambda x: x.todense(), accept_sparse=True)), 
('clf', GradientBoostingRegressor(n_estimators=100,random_state=2)),
])



#create custom scorer
RMSE_scorer = make_scorer(get_RMSE, greater_is_better = False)

##############################################
# Split into Dev and Train data and find best parameters


# Split the data into train data and a dev data based on day of the month.
# This makes sense since the test data is days 19-30 of the month.

	
##############################################
# Create full model using all train data


##############################################
# Create CSV for submission

#.to_csv('preds.csv')

