# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

from sklearn.base import clone
from sklearn.model_selection import KFold
from hp import Hyperparams, ALGS
from score import scoring


def eval_model(model, metric, X, y):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    val_scores = 0
    for train_index, test_index in kf.split(X):
        cloned_model = clone(model)
        cloned_model.fit(X[train_index], y[train_index])
        val_scores += scoring(model=cloned_model, metric=metric, X=X[test_index], y_true=y[test_index])
    val_scores /= n_splits
    return val_scores


class OptunaObjective(object):  # used by optuna
    def __init__(self, algname=None, metric=None, X=None, y=None):
        self.algname = algname
        self.metric = metric
        self.X = X
        self.y = y

        self.alg = ALGS[algname]


    def create_model(self, trial):
        kwargs = {}
        for param in Hyperparams[self.alg]:
            param_name = f'{self.algname}_{param[1]}'
            p = None
            if param[0] == 'categorical':
                p = trial.suggest_categorical(param_name, param[2])
            elif param[0] == 'int':
                p = trial.suggest_int(param_name, param[2], param[3], log=param[4] == 'log')
            elif param[0] == 'float':
                p = trial.suggest_float(param_name, param[2], param[3], log=param[4] == 'log')
            else:
                exit(f'create_model, unknown hyperparameter type: {param[0]}')
            kwargs.update({param[1]: p})
        model = self.alg(**kwargs)
        return model


    def __call__(self, trial):
        model = self.create_model(trial)
        trial_score = eval_model(model, self.metric, self.X, self.y)
        trial.set_user_attr(key='model', value=clone(model))
        return trial_score
# end class Objective


def optuna_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model', value=trial.user_attrs['model'])
