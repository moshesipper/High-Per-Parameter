# hyperparams
# copyright 2022 moshe sipper
# www.moshesipper.com

import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names


# run this function once and retain the dataset names we want to work with
def get_datasets():
    with open('classification.csv', 'w') as f:
        f.write('dataset, samples, features, classes\n')

    for i, classification_dataset in enumerate(classification_dataset_names):
        print(i, classification_dataset, flush=True)
        X, y = fetch_data(classification_dataset, return_X_y=True)
        with open('classification.csv', 'a') as f:
            f.write(f'{classification_dataset}, {X.shape[0]}, {X.shape[1]}, {len(np.unique(y))}\n')


    with open('regression.csv', 'w') as f:
        f.write('dataset, samples, features\n')

    for i, regression_dataset in enumerate(regression_dataset_names):
        print(i, regression_dataset, flush=True)
        X, y = fetch_data(regression_dataset, return_X_y=True)
        with open('regression.csv', 'a') as f:
            f.write(f'{regression_dataset}, {X.shape[0]}, {X.shape[1]}\n')


# classification: retain datasets with <= 10992 samples, <= 100 features
CLF_Datasets =\
    ['Hill_Valley_with_noise', 'Hill_Valley_without_noise', 'movement_libras', 'coil2000', 'mfeat_fourier', 'analcatdata_authorship', 'optdigits', 'mfeat_karhunen', 'splice', 'sonar', 'spambase', 'molecular_biology_promoters', 'mfeat_zernike', 'tokyo1', 'spectf', 'flags', 'texture', 'waveform_40', 'satimage', 'chess', 'kr_vs_kp', 'soybean', 'dermatology', 'ionosphere', 'calendarDOW', 'backache', 'breast_cancer_wisconsin', 'wdbc', 'dis', 'allbp', 'allrep', 'allhyper', 'allhypo', 'hypothyroid', 'auto', 'led24', 'collins', 'agaricus_lepiota', 'mushroom', 'colic', 'horse_colic', 'spect', 'ann_thyroid', 'waveform_21', 'car_evaluation', 'ring', 'twonorm', 'churn', 'GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1', 'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1', 'GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001', 'credit_g', 'german', 'segmentation', 'hepatitis', 'vehicle', 'lymphography', 'pendigits', 'house_votes_84', 'vote', 'labor', 'buggyCrx', 'credit_a', 'crx', 'australian', 'schizo', 'vowel', 'cleve', 'heart_c', 'cleveland', 'heart_h', 'hungarian', 'heart_statlog', 'wine_recognition', 'solar_flare_2', 'solar_flare_1', 'wine_quality_white', 'wine_quality_red', 'analcatdata_fraud', 'page_blocks', 'mofn_3_7_10', 'parity5+5', 'flare', 'breast', 'analcatdata_cyyoung8092', 'analcatdata_cyyoung9302', 'cmc', 'contraceptive', 'xd6', 'tic_tac_toe', 'breast_w', 'profb', 'threeOf9', 'saheart', 'breast_cancer', 'glass', 'prnn_fglass', 'glass2', 'analcatdata_japansolvent', 'yeast', 'diabetes', 'pima', 'cars', 'biomed', 'postoperative_patient_data', 'led7', 'penguins', 'ecoli', 'cleveland_nominal', 'prnn_crabs', 'cloud', 'appendicitis', 'mfeat_morphological', 'car', 'monk2', 'monk1', 'monk3', 'corral', 'mux6', 'analcatdata_creditscore', 'analcatdata_bankruptcy', 'phoneme', 'irish', 'analcatdata_germangss', 'bupa', 'new_thyroid', 'tae', 'parity5', 'analcatdata_dmft', 'balance_scale', 'analcatdata_lawsuit', 'hayes_roth', 'iris', 'analcatdata_aids', 'haberman', 'analcatdata_boxing2', 'analcatdata_boxing1', 'lupus', 'analcatdata_asbestos', 'confidence', 'analcatdata_happiness', 'prnn_synth']


# regression: retain datasets with <= 8192 samples, <= 100 features
REG_Datasets =\
    ['192_vineyard', '228_elusage', '523_analcatdata_neavote', '663_rabe_266', '712_chscase_geyser1', '519_vinnie', 'banana', '678_visualizing_environmental', '556_analcatdata_apnea2', '557_analcatdata_apnea1', 'titanic', '485_analcatdata_vehicle', '1096_FacultySalaries', '690_visualizing_galaxy', '1027_ESL', '1029_LEV', '1030_ERA', '529_pollen', '687_sleuth_ex1605', '594_fri_c2_100_5', '611_fri_c3_100_5', '624_fri_c0_100_5', '656_fri_c1_100_5', '210_cloud', '579_fri_c0_250_5', '596_fri_c2_250_5', '601_fri_c1_250_5', '613_fri_c3_250_5', '597_fri_c2_500_5', '617_fri_c3_500_5', '631_fri_c1_500_5', '649_fri_c0_500_5', '599_fri_c2_1000_5', '609_fri_c0_1000_5', '612_fri_c1_1000_5', '628_fri_c3_1000_5', '706_sleuth_case1202', '665_sleuth_case2002', '230_machine_cpu', '659_sleuth_ex1714', '561_cpu', '522_pm10', '547_no2', '225_puma8NH', '591_fri_c1_100_10', '621_fri_c0_100_10', '634_fri_c2_100_10', '229_pwLinear', '602_fri_c3_250_10', '615_fri_c4_250_10', '635_fri_c0_250_10', '647_fri_c1_250_10', '657_fri_c2_250_10', '604_fri_c4_500_10', '627_fri_c2_500_10', '641_fri_c1_500_10', '646_fri_c3_500_10', '654_fri_c0_500_10', '666_rmftsa_ladata', '1028_SWD', '593_fri_c1_1000_10', '595_fri_c0_1000_10', '606_fri_c2_1000_10', '608_fri_c3_1000_10', '623_fri_c4_1000_10', '695_chatfield_4', '227_cpu_small', '562_cpu_small', '1089_USCrime', '527_analcatdata_election2000', '560_bodyfat', '503_wind', '542_pollution', '195_auto_price', '207_autoPrice', '197_cpu_act', '573_cpu_act', '651_fri_c0_100_25', '605_fri_c2_250_25', '644_fri_c4_250_25', '653_fri_c0_250_25', '658_fri_c3_250_25', '581_fri_c3_500_25', '582_fri_c1_500_25', '584_fri_c4_500_25', '633_fri_c0_500_25', '643_fri_c2_500_25', '586_fri_c3_1000_25', '589_fri_c2_1000_25', '592_fri_c4_1000_25', '598_fri_c0_1000_25', '620_fri_c1_1000_25', '294_satellite_image', '603_fri_c0_250_50', '648_fri_c1_250_50', '616_fri_c4_500_50', '626_fri_c2_500_50', '637_fri_c1_500_50', '645_fri_c3_500_50', '650_fri_c0_500_50', '583_fri_c1_1000_50', '590_fri_c0_1000_50', '607_fri_c4_1000_50', '618_fri_c3_1000_50', '622_fri_c2_1000_50', '588_fri_c4_1000_100']


params = {'figure.dpi': 600,
          'font.sans-serif': 'Calibri',
          'font.weight':  'bold',
          'font.family': 'sans-serif',
          'axes.titlesize': 16,
          'font.size': 12,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'xtick.minor.size': 0,
          'axes.labelsize': 16,
          'legend.fontsize': 16,
          'legend.handlelength': 1,
          'lines.linewidth': 3,
          'lines.markersize': 5}
matplotlib.rcParams.update(params)


def add_commas(l):
    ll = []
    for x in l:
        if x >= 10000:
            ll.append("{:,}".format(x))
        else:
            ll.append(str(x))

    return ll


def plot_datasets(clf_file='csv/clf_retain.csv', reg_file='csv/reg_retain.csv'):
    df = pd.read_csv(clf_file)
    for col_name in df:
        df = df.rename(columns={col_name: col_name.replace(' ', '')})  # remove spaces

    plt.scatter(df['samples'], df['features'])
    plt.title('Classification: Samples vs. Features', fontweight='bold')
    # plt.legend([f'total of {len(df)} classification datasets'])
    plt.xlabel('Samples', fontweight='bold')
    l1 = range(0, max(df['samples']) + 1, 2000)
    l2 = add_commas(l1)
    plt.xticks(ticks=l1, labels=l2)
    plt.ylabel('Features', fontweight='bold')
    plt.show()

    plt.scatter(df['samples'], df['classes'])
    plt.title('Classification: Samples vs. Classes', fontweight='bold')
    # plt.legend([f'total of {len(df)} classification datasets'])
    plt.xlabel('Samples', fontweight='bold')
    l1 = range(0, max(df['samples']) + 1, 2000)
    l2 = add_commas(l1)
    plt.xticks(ticks=l1, labels=l2)
    minimum_ele = min(df['classes'])
    maximum_ele = max(df['classes'])
    new_list = range(math.floor(minimum_ele), math.ceil(maximum_ele) + 1, 2)
    plt.yticks(new_list)
    plt.ylabel('Classes', fontweight='bold')
    plt.show()

    plt.scatter(df['features'], df['classes'])
    plt.title('Classification: Features vs. Classes', fontweight='bold')
    # plt.legend([f'total of {len(df)} classification datasets'])
    plt.xlabel('Features', fontweight='bold')
    minimum_ele = min(df['classes'])
    maximum_ele = max(df['classes'])
    new_list = range(math.floor(minimum_ele), math.ceil(maximum_ele) + 1, 2)
    plt.yticks(new_list)
    plt.ylabel('Classes', fontweight='bold')
    plt.show()

    df = pd.read_csv(reg_file)
    for col_name in df:
        df = df.rename(columns={col_name: col_name.replace(' ', '')})  # remove spaces

    plt.scatter(df['samples'], df['features'])
    plt.title('Regression: Samples vs. Features', fontweight='bold')
    # plt.legend([f'total of {len(df)} regression datasets'])
    plt.xlabel('Samples', fontweight='bold')
    plt.ylabel('Features', fontweight='bold')
    plt.show()
