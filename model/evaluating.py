from sklearn.metrics import log_loss, roc_auc_score

__all__ = ["evaluating"]

def evaluating(model, test_model_input, test):
    preds = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test['target'].values, preds), 4))
    print("test AUC", round(roc_auc_score(test['target'].values, preds), 4))

    return preds