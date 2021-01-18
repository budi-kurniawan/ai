#!/usr/bin/env python
# coding: utf-8

# In[2]:


print('--- import ---')
import numpy as np
import pandas as pd

''' data cleanning '''
import texthero as hero
from texthero import preprocessing

''' model building and evaluation '''
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

''' Evaluation '''
from yellowbrick.model_selection import LearningCurve
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ClassificationReport

''' Plotting '''
import matplotlib
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_learning_curves

print('--- get file ---')
# columns that we use
cols = ['EventDescription','IncidentCause','IncidentConsequence','Category']

# importing file
df = pd.read_csv('../../cleaned_incidents1.csv', usecols=cols)

# drop missing category
df = df.dropna(axis=0, subset=['Category'])

# replace the rest with empty string
df = df.replace(np.nan, '', regex=True)

print('--- declare function ---')
# Text Cleaning and Pre-processing
def preprocess_text(features):
    # cleaning steps
    cleaning_pipeline = [
        preprocessing.fillna,
        preprocessing.lowercase,
        preprocessing.remove_whitespace,
        preprocessing.remove_punctuation,
        preprocessing.remove_urls,
        preprocessing.remove_brackets,
        preprocessing.remove_stopwords,
        preprocessing.remove_digits,
        preprocessing.remove_angle_brackets,
        preprocessing.remove_curly_brackets,
        preprocessing.stem
    ]

    # apply pipeline to text
    clean_text = features.pipe(hero.clean, cleaning_pipeline)
    
    return clean_text

print('--- cleaning data ---')
# --- Cleaning data ---
# cleaning the data
df['description'] = df['EventDescription'] + ' ' + df['IncidentCause']+ ' ' + df['IncidentConsequence']
df['description'] = preprocess_text(df['description'])

le = LabelEncoder()
Y = le.fit_transform(df['Category'])

# splitting of data in test and train
x_train,x_test, y_train, y_test = train_test_split(df['description'],Y, test_size=0.25, random_state = 42)

# vectorize
tfidf = TfidfVectorizer(analyzer='word', max_features=500)

tfidf.fit_transform(df['description']).toarray()
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)

# resampling
oversample = SMOTE(random_state=0,n_jobs=-1,k_neighbors=5)
x_train, y_train = oversample.fit_resample(x_train, y_train)

# --- Optimisation ---
print('--- Optimisation ---')
param_grid = {
    'learning_rate': [0.001, 0.10, 0.30],
    'max_depth' : [6, 10, 20, 30],
    'gamma' : [0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0]
}
best_model = GridSearchCV(estimator=XGBClassifier(random_state=4),
                      param_grid=param_grid, cv= 5, n_jobs=-1)
best_model.fit(x_train, y_train)
print(best_model.best_params_)
print('finish')


# In[ ]:




