import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

def get_train_test_df(filename):
    """
    Get Train Test dataframes
    """
    df=pd.read_csv(filename,header=0)
    X,y=df.drop(['target'],axis=1),df['target']
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_preprocess_transformer(dataframe):
    coltrf=ColumnTransformer(
    [('scaler',StandardScaler(),['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
     ('onehot',OneHotEncoder(),['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ],
    remainder='passthrough'
    )
    return coltrf.fit(dataframe)
    
def train_model(X,y):
    """
    Train Model
    """
    lr=LogisticRegression()
    lr.fit(X,y)
    return lr

def evaluate_model(model,X,y):
    """
    Evaluate Model
    """
    return model.score(X,y)
    
if __name__=='__main__':
    # Train
    X_train, X_test, y_train, y_test=get_train_test_df('dataset.csv')
    transformer_preprocess=get_preprocess_transformer(X_train)
    X_train=transformer_preprocess.transform(X_train)
    model=train_model(X_train,y_train)
    train_score=evaluate_model(model,X_train,y_train)
    #Test
    X_test=transformer_preprocess.transform(X_test)
    test_score=cross_val_score(model, X_test,y_test, cv=3)
    #Output
    print(f'TrainScore:{train_score}    TestScore:{test_score.mean()}')
    