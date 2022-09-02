# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

import re
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier, SGDRegressor, \
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, BayesianRidge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

Classifiers = [LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, LinearSVC,
               DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
               KNeighborsClassifier, MultinomialNB, XGBClassifier, LGBMClassifier]
CLFS = dict(zip([k.__name__ for k in Classifiers], Classifiers))

# Hyperparameter ranges/sets for the ML algorithms
# Also, for Optuna, the hyperparameter type: 'categorical', 'int', or 'float'
Hyperparams = {
    AdaBoostClassifier: [['int', 'n_estimators', 10, 1000, 'log'],
                         ['float', 'learning_rate', 0.1, 10, 'log']],

    AdaBoostRegressor: [['int', 'n_estimators', 10, 1000, 'log'],
                        ['float', 'learning_rate', 0.1, 10, 'log']],

    BayesianRidge: [['int', 'n_iter', 10, 1000, 'log'],
                    ['float', 'alpha_1', 1e-7, 1e-5, 'log'],
                    ['float', 'lambda_1', 1e-7, 1e-5, 'log'],
                    ['float', 'tol', 1e-5, 1e-1, 'log']],

    DecisionTreeClassifier: [['int', 'max_depth', 2, 10, 'nolog'],
                             ['float', 'min_impurity_decrease', 0., 0.5, 'nolog'],
                             ['categorical', 'criterion', ['gini', 'entropy']]],

    DecisionTreeRegressor: [['int', 'max_depth', 2, 10, 'nolog'],
                            ['float', 'min_impurity_decrease', 0., 0.5, 'nolog'],
                            ['categorical', 'criterion', ['squared_error', 'friedman_mse', 'absolute_error']]],

    GradientBoostingClassifier: [['int', 'n_estimators', 10, 1000, 'log'],
                                 ['float', 'learning_rate', 0.01, 0.3, 'nolog'],
                                 ['float', 'subsample', 0.1, 1, 'nolog']],

    GradientBoostingRegressor: [['int', 'n_estimators', 10, 1000, 'log'],
                                ['float', 'learning_rate', 0.01, 0.3, 'nolog'],
                                ['float', 'subsample', 0.1, 1, 'nolog']],

    KNeighborsClassifier: [['categorical', 'weights', ['uniform', 'distance']],
                           ['categorical', 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']],
                           ['int', 'n_neighbors', 2, 20, 'nolog']],

    KNeighborsRegressor: [['categorical', 'weights', ['uniform', 'distance']],
                          ['categorical', 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']],
                          ['int', 'n_neighbors', 2, 20, 'nolog']],

    KernelRidge: [['categorical', 'kernel', ['linear', 'poly', 'rbf', 'sigmoid']],
                  ['float', 'alpha', 0.1, 10, 'log'],
                  ['float', 'gamma', 0.1, 10, 'log']],

    LGBMClassifier: [['int', 'n_estimators', 10, 1000, 'log'],
                     ['float', 'learning_rate', 0.01, 0.2, 'nolog'],
                     ['float', 'bagging_fraction', 0.5, 0.95, 'nolog']],

    LGBMRegressor: [['float', 'lambda_l1', 1e-8, 10.0, 'log'],
                    ['float', 'lambda_l2', 1e-8, 10.0, 'log'],
                    ['int', 'num_leaves', 2, 256, 'nolog']],

    LinearRegression: [['categorical', 'fit_intercept', [True, False]],
                       ['categorical', 'normalize', [True, False]]],

    LinearSVC: [['int', 'max_iter', 10, 10000, 'log'],
                ['float', 'tol', 1e-5, 1e-1, 'log'],
                ['float', 'C', 0.01, 10, 'log']],

    LinearSVR: [['categorical', 'loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']],
                ['float', 'tol', 1e-5, 1e-1, 'log'],
                ['float', 'C', 0.01, 10, 'log']],

    LogisticRegression: [['categorical', 'penalty', ['l1', 'l2']],
                         ['categorical', 'solver', ['liblinear', 'saga']]],

    MultinomialNB: [['float', 'alpha', 0.01, 10, 'log'],
                    ['categorical', 'fit_prior', [True, False]]],

    PassiveAggressiveClassifier: [['float', 'C', 1e-2, 10, 'log'],
                                  ['categorical', 'fit_intercept', [True, False]],
                                  ['int', 'max_iter', 10, 1000, 'log']],

    PassiveAggressiveRegressor: [['float', 'C', 1e-2, 10, 'log'],
                                 ['categorical', 'fit_intercept', [True, False]],
                                 ['int', 'max_iter', 10, 1000, 'log']],

    RandomForestClassifier: [['int', 'n_estimators', 10, 1000, 'log'],
                             ['float', 'min_weight_fraction_leaf', 0., 0.5, 'nolog'],
                             ['categorical', 'max_features', ['auto', 'sqrt', 'log2']]],

    RandomForestRegressor: [['int', 'n_estimators', 10, 1000, 'log'],
                            ['float', 'min_weight_fraction_leaf', 0., 0.5, 'nolog'],
                            ['categorical', 'max_features', ['auto', 'sqrt', 'log2']]],

    RidgeClassifier: [['categorical', 'solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']],
                      ['float', 'alpha', 1e-3, 10, 'log']],

    SGDClassifier: [['categorical', 'penalty', ['l2', 'l1', 'elasticnet']],
                    ['float', 'alpha', 1e-5, 1, 'log']],

    SGDRegressor: [['float', 'alpha', 1e-05, 1, 'log'],
                   ['categorical', 'penalty', ['l2', 'l1', 'elasticnet']]],

    XGBClassifier: [['int', 'n_estimators', 10, 1000, 'log'],
                    ['float', 'learning_rate', 0.01, 0.2, 'nolog'],
                    ['float', 'gamma', 0., 0.4, 'nolog']],

    XGBRegressor: [['int', 'n_estimators', 10, 1000, 'log'],
                   ['float', 'learning_rate', 0.01, 0.2, 'nolog'],
                   ['float', 'gamma', 0., 0.4, 'nolog']],

}

ALGS = dict(zip([k.__name__ for k in Hyperparams.keys()], list(Hyperparams.keys())))


'''
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py
def is_classifier(estimator):
    """Return True if the given estimator is (probably) a classifier.
    Parameters
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"
'''


def is_classifier(model):
    if isinstance(model, str):
        return model in list(CLFS.keys())
    else:
        try:
            if model in Hyperparams.keys():
                return model in CLFS.values()
            else:
                return model.__class__ in CLFS.values()
        except:
            return model.__class__ in CLFS.values()


def format_param(s):
    if isinstance(s, bool):
        return str(s)
    else:
        return re.sub('_', '\\_', s)


def latex_hp(clf=True):
    # generate latex table of hyperparams, either for classifiers or for regressors
    print('    \\begin{tabular}{r|c|l}')
    print('    \\hline')
    print('    \\textbf{Algorithm} & \\textbf{Hyperparameter} & \\textbf{Values} \\\\ \\hline')
    for key, val in Hyperparams.items():
        if (clf and is_classifier(key)) or (not clf and not is_classifier(key)):
            first = True
            for param in val:
                algname = f'\\multirow{{{len(val)}}}*{{{key.__name__}}}' if first else ''
                first = False
                param_name = format_param(param[1])
                if param[0] == 'categorical':
                    params = '\\{'
                    for s in param[2]:
                        params += format_param(s) + ', '
                    params = re.sub(', $', '\\}', params)
                    print(f'    {algname} & {param_name} & {params} \\\\')
                elif param[0] == 'int' or param[0] == 'float':
                    log = ' (log)' if param[4] == 'log' else ''
                    print(f'    {algname} & {param_name} & [{param[2]}, {param[3]}] {log} \\\\')
                else:
                    exit(f'error in HyperParams dict: {param[0]}')
            print('    \\hline')
    print('    \\end{tabular}')
