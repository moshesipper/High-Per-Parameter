# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_squared_error
from hp import is_classifier


MetricNames = {'clf': {'metric1': 'Acc', 'metric2': 'Bal', 'metric3': 'F1'},
               'reg': {'metric1': 'R2', 'metric2': 'Adj R2', 'metric3': 'C-RMSE'}}


def metric_name(algname, metric):
    return MetricNames['clf' if is_classifier(algname) else 'reg'][metric]


def adjusted_r2(r2, n, p):
    # r2: r2 score
    # n: number of samples
    # p: number of features
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def scoring(model, metric, X, y_true):
    assert metric in ['metric1', 'metric2', 'metric3'], f"unknown metric type in function `scoring': {metric}"

    y_pred = model.predict(X)

    if is_classifier(model):  # classification
        if metric == 'metric1':
            return accuracy_score(y_true, y_pred)
        elif metric == 'metric2':
            return balanced_accuracy_score(y_true, y_pred)
        elif metric == 'metric3':
            return f1_score(y_true, y_pred, average='macro')
    else:  # regression
        if metric == 'metric1':
            return r2_score(y_true, y_pred)
        elif metric == 'metric2':
            r2 = r2_score(y_true, y_pred)
            return adjusted_r2(r2, X.shape[0], X.shape[1])
        elif metric == 'metric3':
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            complement_rmse = 1 - rmse
            return complement_rmse
