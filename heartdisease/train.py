import warnings

import numpy as np
import pandas as pd
from helper import get_feature_names
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


def get_train_test_df(filename):
    """
    Get Train Test dataframes
    """
    df = pd.read_csv(filename, header=0)
    X, y = df.drop(['target'], axis=1), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_preprocess_transformer(dataframe):
    coltrf = ColumnTransformer(
    [('scaler', StandardScaler(), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
     ('onehot', OneHotEncoder(), [
      'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ],
    remainder='passthrough'
    )
    return coltrf.fit(dataframe)


def train_model(X, y):
    """
    Train Model
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr


def find_best_model(X, y,transformed_features):
    """
    Search Grid for Optimal # of Features and Best Classifier
    """
    pipe = pipeline.Pipeline(
        [
            ('feature_selector', SelectKBest(mutual_info_classif, k=8)),
            ('classifier', LogisticRegression())
        ]
    )
    search_space=[
        {
        'feature_selector__k':[3,5,8,11]
        },
        {
        'classifier':[LogisticRegression()],
        'classifier__C':[0.1,0.3,0.5,1]
        },
        {
        'classifier':[RandomForestClassifier(n_estimators=100)],
        'classifier__max_depth':[5,10,None]
        },
        {
        'classifier':[KNeighborsClassifier()],
        'classifier__n_neighbors':[3,7,11],
        'classifier__weights':['uniform','distance']
        }
    ]
    cv=GridSearchCV(pipe,search_space,cv=10)
    cv.fit(X,y)
    print(f'Best Estimator:\n {cv.best_estimator_}')
    best_features=cv.best_estimator_[0].get_support()
    transformed_features=pd.Index(transformed_features)
    best_features=transformed_features[best_features]
    print(f'Best Features:\n {best_features}')
    print(f'Best Cross Validated Train Score:\n {cv.best_score_}')
    return cv


def evaluate_model(model, X, y):
    """
    Evaluate Model
    """
    return model.score(X, y)

if __name__ == '__main__':
    # Train
    X_train, X_test, y_train, y_test=get_train_test_df('dataset.csv')
    transformer_preprocess=get_preprocess_transformer(X_train)
    X_train=transformer_preprocess.transform(X_train)
    # model=train_model(X_train, y_train)
    transformed_features=get_feature_names(transformer_preprocess)
    model=find_best_model(X_train, y_train,transformed_features)
    train_score=evaluate_model(model, X_train, y_train)
    # Test
    X_test=transformer_preprocess.transform(X_test)
    test_score=cross_val_score(model, X_test, y_test, cv=3)
    # Output
    print(f'TrainScore:{train_score}    \nTestScore:{test_score.mean()}')
