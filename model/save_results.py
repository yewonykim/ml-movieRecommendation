import pandas as pd
from sklearn.preprocessing import LabelEncoder

__all__ = ["save_results"]

def save_results(test, preds, path):
    test['pred_target'] = preds
    original_data = pd.read_csv(path)
    lbe = LabelEncoder()
    lbe.fit(original_data['title'])
    test['title'] = lbe.inverse_transform(test['title'])
    test[['userId', 'title', 'pred_target']].to_csv('./decoded_predictions.csv', index=False)