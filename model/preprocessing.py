import pandas as pd
from sklearn.preprocessing import LabelEncoder

__all__ = ["preprocessing"]

def preprocessing(path):
    df = pd.read_csv(path)

    sparse_features = ['userId', 'title', 'genres', 'tag']

    label_encoders = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
        label_encoders[feat] = lbe

    return df,  sparse_features