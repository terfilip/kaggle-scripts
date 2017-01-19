
# coding: utf-8

# # Kaggle Titanic

# In[27]:

import numpy as np
import pandas as pd
import re
import csv
import matplotlib.pyplot as plt
import sys
import statsmodels.imputation.mice as mice

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from statistics import mode

TEST_BEGIN_ROW = 891
TRAIN_CSV = 'input/train.csv'
TEST_CSV = 'input/test.csv'

def read_and_merge():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    return pd.concat([train, test])

def split(merged):
    return (merged[:TEST_BEGIN_ROW], merged[TEST_BEGIN_ROW:])

def is_mother(row):
    if (row['Sex_female'] == 1) and (row['Age'] > 18) and (row['Parch'] > 1):
        return 1
    else:
        return 0

def is_child(row):
    if row['Age'] < 18:
        return 1
    else:
        return 0

def get_title(row):
    match = re.findall(r'.,\s(\S+)\..', row['Name'])
    if len(match) > 0:
        title = match[0]
        if title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Mme':
            return 'Mrs'
        elif title in ['Dona', 'Lady', 'Sir',
                       'Dr', 'Jonkheer', 'Don', 'Rev']:
            return 'Noble'
        elif title in ['Major', 'Capt', 'Col', 'Master']:
            return 'Military'
        else:
            return title
    else:
        return ''

def clean_age_predictions(predicted):
    for i in range(len(predicted)):
        predicted[i] = abs(round(predicted[i], 0))
        if predicted[i] < 0:
            predicted[i] = 1

def extract_deck(row):
    if pd.isnull(row['Cabin']):
        return row['Cabin']
    return row['Cabin'][0]

def enum_deck(row):
    if pd.isnull(row['Deck']):
        return row['Deck']
    return ord(row['Deck']) - 65

def fill_ages(merged):
    #Don't bother with features that are not numeric for predicting age + Survived
    merged_numeric = merged.drop(['Deck', 'Survived', 'Surname', 'PassengerId'],
                                  axis=1)
    imp = mice.MICEData(merged_numeric)
    imp.perturb_params('Age')
    imp.impute_pmm('Age')
    return imp.data.Age
    
def fill_decks(data):
    
    redundant_cols = ['Surname', 'Survived', 'Deck']
    deck_train_y = data.loc[data.Deck.notnull()]['Deck']
    deck_train_x = data.loc[data.Deck.notnull()]                        .drop(redundant_cols, axis=1)
    
    deck_test = data.loc[data.Deck.isnull()]                     .drop(redundant_cols, axis=1)
        
    alg = RandomForestClassifier(n_estimators=1000)
    model = alg.fit(deck_train_x, deck_train_y)
    data.ix[data.Deck.isnull(), 'Deck'] = model.predict(deck_test)

merged = read_and_merge()

merged['Surname'] = merged.apply(lambda row: row['Name'].split(',')[0], axis=1)
merged['Title'] = merged.apply(get_title, axis=1)
merged['Cabin'] = merged.apply(extract_deck, axis=1)
merged.rename(columns={'Cabin' : 'Deck'}, inplace=True)
merged.drop(['Name','Ticket'], axis=1, inplace=True)

embarked_mode = mode(merged['Embarked'])
merged.set_value(61, 'Embarked', embarked_mode)
merged.set_value(829, 'Embarked', embarked_mode) 
fare_mean = round(np.mean(merged.Fare), 2)
merged.ix[merged.Fare.isnull(), 'Fare'] = fare_mean
merged = pd.get_dummies(merged, columns=['Title', 'Embarked', 'Pclass', 'Sex'])
merged['Deck'] = merged.apply(enum_deck, axis=1)
merged.Age = fill_ages(merged)

merged['Mother'] = merged.apply(is_mother, axis=1)
merged['Child'] = merged.apply(is_child, axis=1)
#merged = pd.get_dummies(merged, columns=['Surname'])
#merged.drop(['Surname', 'Deck'], 1, inplace=True)

def fill_family_deck(group):
     if (not all(group.isnull())) and any(group.isnull()):
        return group.fillna(method='ffill').fillna(method='bfill')
     else:
        return group

merged.Deck = merged.groupby('Surname')['Deck'].transform(fill_family_deck)
fill_decks(merged)
merged = pd.get_dummies(merged, columns=['Surname'])
forest = RandomForestClassifier(n_estimators=3000)
train, test = split(merged)
model = forest.fit(train.drop('Survived', 1), train.Survived)
res = model.predict(test.drop('Survived', 1))

results = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived' : res.astype(int)}).set_index('PassengerId')
results.to_csv('output/res.csv')




