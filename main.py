from model import preprocessing
from model import modeling
from model import evaluating
from model import save_results

def total(path):
    df, sparse_features = preprocessing(path)
    model, test_model_input, test = modeling(df, sparse_features)
    preds = evaluating(model, test_model_input, test)
    save_results(test, preds, path)

path = "./movielens.csv"
total(path)