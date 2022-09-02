# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

import pandas as pd
import os
from statistics import mean
from sklearn.preprocessing import RobustScaler
from hp import is_classifier
results_dir = r'./results'


def fmt(x):
    if abs(x) > 1000:
        return f'{x:.1e}'
    else:
        return f'{x:.1f}'


print('\n' * 80)
total_replicates = 0
metrics = dict()
metrics['clf'], metrics['reg'] = [], []
for key in ['clf', 'reg']:
    print()
    for filename in sorted(os.listdir(results_dir)):
        f = os.path.join(results_dir, filename)
        df = pd.read_csv(f, header=None, on_bad_lines='skip')
        df = df.dropna(axis=0)
        algname = df[0][0]
        if (key == 'clf' and is_classifier(algname)) or (key == 'reg' and not is_classifier(algname)):
            total_replicates += df.shape[0]
            metric1_median, metric1_mean, metric1_std = df[2].median(), df[2].mean(), df[2].std()
            metric2_median, metric2_mean, metric2_std = df[3].median(), df[3].mean(), df[3].std()
            metric3_median, metric3_mean, metric3_std = df[4].median(), df[4].mean(), df[4].std()
            print(f'         {algname} & '
                  f'{fmt(metric1_median)} & {fmt(metric1_mean)} ({fmt(metric1_std)}) & '
                  f'{fmt(metric2_median)} & {fmt(metric2_mean)} ({fmt(metric2_std)}) & '
                  f'{fmt(metric3_median)} & {fmt(metric3_mean)} ({fmt(metric3_std)}) & {df.shape[0]} \\\\ \\hline')
            metrics[key].append([algname,
                                 metric1_median, metric1_mean, metric1_std,
                                 metric2_median, metric2_mean, metric2_std,
                                 metric3_median, metric3_mean, metric3_std])

print()
print(f'maximal possible total: {144*30*13 + 106*30*13}')
print(f'\\newcommand\\reps{{{total_replicates:,d} }}')
print(f'\\newcommand\\runs{{{300*total_replicates:,d} }}')

for key in ['clf', 'reg']:
    print()
    hp_scores = dict()
    metrs = [m[1:] for m in metrics[key]]
    scaled_metrics = RobustScaler().fit_transform(metrs)
    for i, alg in enumerate(metrics[key]):
        algname = alg[0]
        hp_scores[algname] = mean(scaled_metrics[i])
    srt = sorted(hp_scores.items(), key=lambda x: x[1], reverse=True)
    for alg in srt:
        print(f'         {alg[0]} & {alg[1]:.2f} \\\\')
