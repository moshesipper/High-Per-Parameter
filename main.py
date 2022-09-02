# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

# Use three score metrics, 'metric1', 'metric2', 'metric3', which are, respectively:
#   - For classification: accuracy, balanced accuracy, f1
#   - For regression: r2, adjusted r2, complement rmse

import argparse
import warnings
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pmlb import fetch_data

from hp import ALGS, is_classifier
from datasets import CLF_Datasets, REG_Datasets
from tune import OptunaObjective, optuna_callback, eval_model
from score import scoring

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)  # verbosity levels: optuna.readthedocs.io/en/stable/reference/logging.html

n_replicates = 30  # number of replicates
n_trials = 50  # number of Optuna trials, also number of runs with default values
time_limit = 259200  # 72 hours, limit for all replicates, 72*60*60 = 259200
datasets_dir = '../datasets/pmlb'
results_dir = 'results'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, action='store', default=None,
                        help=f'ML algorithm to use (default: None)')
    parser.add_argument('--dataset', type=str, action='store', default=None,
                        help='Dataset to use (default: None)')
    args = parser.parse_args()
    return args.alg, args.dataset


# main
def main():
    algname, dataset = get_args()
    assert algname in ALGS, f'unknown algorithm {algname}'
    assert dataset in CLF_Datasets + REG_Datasets, f'unknown dataset {dataset}'
    alg_type = 'classification' if is_classifier(algname) else 'regression'
    print(algname, dataset, alg_type)

    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir=datasets_dir)
    if algname == 'XGBClassifier':
        # To avoid this type of error:
        #     Invalid classes inferred from unique values of `y`. Expected: [0 1], got [1 2]
        le = LabelEncoder()
        y = le.fit_transform(y)

    tic_all = time.time()
    for rep in range(1, n_replicates + 1):
        tic_rep = time.time()
        print('rep', rep, flush=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)  # scale training data
        X_test = scaler.transform(X_test)  # use SAME scaler as one fitted to training data

        opt = {'metric1': -1, 'metric2': -1, 'metric3': -1}  # Optuna run scores per metric
        dfl = {'metric1': -1, 'metric2': -1, 'metric3': -1}  # Default-hyperparam run scores per metric
        imp = {'metric1': -1, 'metric2': -1, 'metric3': -1}  # Improvement percentages per metric
        for metric in ['metric1', 'metric2', 'metric3']:
            # find the best hyperparameters over the training set using Optuna
            objective = OptunaObjective(algname=algname, metric=metric, X=X_train, y=y_train)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
            best_model = study.user_attrs['best_model']
            best_model.fit(X_train, y_train)
            opt[metric] = scoring(model=best_model, metric=metric, X=X_test, y_true=y_test)  # test-set score

            # find the best hyperparameters over the training set using default
            best_model, best_score = None, float('-inf')
            for i in range(n_trials):  # run default hyperparams for same number of trials as Optuna
                model = ALGS[algname]()
                score = eval_model(model=model, metric=metric, X=X_train, y=y_train)
                if score > best_score:
                    best_score = score
                    best_model = clone(model)
            best_model.fit(X_train, y_train)
            dfl[metric] = scoring(model=best_model, metric=metric, X=X_test, y_true=y_test)  # test-set score

        dfl1 = 0.001 if abs(dfl['metric1']) < 1e-09 else abs(dfl['metric1'])
        dfl2 = 0.001 if abs(dfl['metric2']) < 1e-09 else abs(dfl['metric2'])
        dfl3 = 0.001 if abs(dfl['metric3']) < 1e-09 else abs(dfl['metric3'])
        imp['metric1'] = (opt['metric1'] - dfl['metric1']) / dfl1 * 100
        imp['metric2'] = (opt['metric2'] - dfl['metric2']) / dfl2 * 100
        imp['metric3'] = (opt['metric3'] - dfl['metric3']) / dfl3 * 100
        with open(f'{results_dir}/{algname}.csv', 'a') as f:
            print(f"{algname}, {dataset}, "
                  f"{imp['metric1']}, {imp['metric2']}, {imp['metric3']}, "
                  f"{opt['metric1']}, {opt['metric2']}, {opt['metric3']}, "
                  f"{dfl['metric1']}, {dfl['metric2']}, {dfl['metric3']}",
                  file=f)

        toc = time.time()
        with open('times.csv', 'a') as f:
            print(f'{rep}, {algname}, {dataset}, {toc - tic_rep:.0f}', file=f)
        if (toc - tic_all) > time_limit:
            with open('timeout.csv', 'a') as f:
                print(f'{rep}, {algname}, {dataset}, {toc - tic_all:.0f}', file=f)
            break


##############
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
