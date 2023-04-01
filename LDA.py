import os
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import define_distribution
import delonge
import files
import generation
import plots

GENERATED_SAMPLE_DIRECTORY = 'csv_files'
CRITERIA_RESULTS_DIRECTORY = 'criteria_results'
ROC_DISTRIBUTION_DIRECTORY = 'roc_distribution'
DELONG_PATH = 'DeLonge/'

MODEL_PARAMETERS_FILE = './input_data/input.txt'
GENERATED_SAMPLE_FILE = 'csv_files/generated_sample.csv'
EXPERIMENT_PARAMETERS_FILE = 'input_data/params.txt'
FREQUENCY_NORMAL_FILE = 'input_data/frequency_normal.txt'
FREQUENCY_ANEMIA_FILE = 'input_data/frequency_anemia.txt'

DISTRIBUTIONS = ['norm']  # , 'laplace', 'uniform', 'logistic', 'gumbel']
SIMULATION_AMOUNT = [100]
STD_RATIO_NORMAL = [0.032]  # 0.008, 0.015, 0.026, 0.032, 0.04, 0.0455, 0.05, 0.083]
STD_RATIO_ANEMIA = [0.059]  # 0.083, 0.076, 0.065, 0.059, 0.051, 0.0455, 0.041, 0.008]

NORMAL_OBSERVATIONS_WEEKS_START = 11
NORMAL_OBSERVATIONS_WEEKS_END = 34
ANEMIA_OBSERVATIONS_WEEKS_START = 15
ANEMIA_OBSERVATIONS_WEEKS_END = 34

APRIOR_PROBABILITY_NORMAL = 0.8
APRIOR_PROBABILITY_ANEMIA = 1 - APRIOR_PROBABILITY_NORMAL

SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_LEVEL = 0.95


@dataclass
class ModelCTR:
    free_coef: float
    week_coef: float
    std: float

    def calculate_ctr_index(self, week, noise):
        # генерация КТИ
        return self.free_coef + self.week_coef * week + noise


@dataclass
class ExperimentParameters:
    normal_data_amount: int
    anemia_data_amount: int
    train_rate: float
    find_roc_distribution: int
    do_delonge: int


def main_process():
    # если существует файл с записями, очистить его
    files.create_dir(GENERATED_SAMPLE_DIRECTORY)
    files.create_dir(CRITERIA_RESULTS_DIRECTORY)
    files.clear_file(GENERATED_SAMPLE_FILE)



    model_parameters = [float(i) for i in files.read_parameters(MODEL_PARAMETERS_FILE)]
    normal_model_ctr = ModelCTR(model_parameters[0], model_parameters[1], model_parameters[2])
    anemia_model_ctr = ModelCTR(model_parameters[3], model_parameters[4], model_parameters[5])

    #  считываем параметры генерации и присваиваем переменнным соответствующие значения
    parameters = [i for i in files.read_parameters(EXPERIMENT_PARAMETERS_FILE)]

    experiment_parameters = ExperimentParameters(int(parameters[0]), int(parameters[1]),
                                                 float(parameters[2]),
                                                 int(parameters[3]), int(parameters[4]))

    # создаем интервалы с вероятностными частотами
    intervals_norm = generation.create_intervals(FREQUENCY_NORMAL_FILE)
    intervals_anem = generation.create_intervals(FREQUENCY_ANEMIA_FILE)

    # создали недели, на которых могут проводиться измерения
    # нормальные с 11 недели
    weeks_normal = range(NORMAL_OBSERVATIONS_WEEKS_START, NORMAL_OBSERVATIONS_WEEKS_END)
    # анемия с 15 недели
    weeks_anemia = range(ANEMIA_OBSERVATIONS_WEEKS_START, ANEMIA_OBSERVATIONS_WEEKS_END)
    # делаем каждому элементу пару - будущее количество измерений на данной неделе
    weeks_normal = [[x] + [0] for x in weeks_normal]
    weeks_anemia = [[x] + [0] for x in weeks_anemia]

    for sigma_ratio_normal, sigma_ratio_anemia in zip(STD_RATIO_NORMAL, STD_RATIO_ANEMIA):
        sigma_ratio = round(sigma_ratio_normal / sigma_ratio_anemia, 2)
        for simulation_amount in SIMULATION_AMOUNT:
            for distribution in DISTRIBUTIONS:
                spec_list = []
                sens_list = []
                roc_list = []
                if experiment_parameters.do_delonge:
                    z_score_list = []
                    count = 0
                for i in range(simulation_amount):
                    xy = generation.generate_sample(distribution,
                                                    normal_model_ctr, anemia_model_ctr,
                                                    weeks_normal, weeks_anemia,
                                                    intervals_norm, intervals_anem,
                                                    experiment_parameters.normal_data_amount,
                                                    experiment_parameters.anemia_data_amount,
                                                    GENERATED_SAMPLE_FILE)

                    x_train, x_test, y_train, y_test = train_test_split(xy[:, :2], xy[:, 2],
                                                                        test_size=(
                                                                                1 - experiment_parameters.train_rate),
                                                                        random_state=np.random.randint(10000))

                    lda = LinearDiscriminantAnalysis(solver='svd', priors=[APRIOR_PROBABILITY_NORMAL,
                                                                           APRIOR_PROBABILITY_ANEMIA])
                    lda.fit(x_train, y_train)
                    plots.scatter_plot(x_test, y_test, lda.coef_[0][0], lda.coef_[0][1], lda.intercept_)

                    if experiment_parameters.do_delonge:
                        files.create_dir(DELONG_PATH)
                        lda_eigen = LinearDiscriminantAnalysis(solver='eigen', priors=[APRIOR_PROBABILITY_NORMAL,
                                                                                       APRIOR_PROBABILITY_ANEMIA])
                        lda_eigen.fit(x_train, y_train)

                    # вычисление специфичности и чувствительности
                    y_predict = lda.predict(x_test)
                    specificity, sensitivity = clf_indicators(y_test, y_predict)

                    y_score = lda.decision_function(x_test)
                    fpr, tpr, roc_auc, thresholds = compute_roc(y_test, y_score)
                    # plots.plot_roc(fpr, tpr, roc_auc)

                    # получаем чувствительность и специфичность и ROC
                    # добавляем результаты в списки и повторяем процесс
                    spec_list.append(specificity)
                    sens_list.append(sensitivity)
                    roc_list.append(roc_auc)

                # сохранение данных после циклов по исследованию отклонения от нормальности
                filepath_criteria = CRITERIA_RESULTS_DIRECTORY + '/' + str(simulation_amount) + '_' + str(sigma_ratio)
                csv_filepath_criteria = filepath_criteria + '.csv'
                xlsx_filepath_criteria = filepath_criteria + '.xlsx'

                get_agregated_results_to_file(csv_filepath_criteria, distribution, spec_list, sens_list, roc_list)
                df = pd.read_csv(csv_filepath_criteria)
                df.to_excel(xlsx_filepath_criteria, index=None, header=True)

                if experiment_parameters.find_roc_distribution:
                    files.create_dir(ROC_DISTRIBUTION_DIRECTORY)

                    dist = define_distribution.get_best_distribution(ROC_DISTRIBUTION_DIRECTORY,
                                                                     distribution,
                                                                     roc_list,
                                                                     sigma_ratio,
                                                                     simulation_amount)

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        pd.set_option("colheader_justify", "left")
                        print(f"Наилучшее распределение: {dist}")

                if experiment_parameters.do_delonge:
                    # Блок для ДеЛонге, Вычисление Z-оценок, p-vaue, подбор теоретической
                    # функции нормального распределения для эмпирического распределения Z-оценок
                    # log10(p)
                    p_delong, z_score = delonge.delonge_roc_test(y_test,
                                                                 lda_eigen.predict_proba(x_test)[:, 1],
                                                                 lda_eigen.predict_proba(x_test)[:, 1])
                    # для two-tailed нужно взять экспоненту
                    # p_delong = pow(10, p_delong)
                    z_score_list.append(z_score[0][0])
                    if p_delong > SIGNIFICANCE_LEVEL:
                        count += 1

            if experiment_parameters.do_delonge:
                z_score_list.sort()

                # общее название файла для запуска
                filename_delonge = DELONG_PATH + str(sigma_ratio) + '_' + str(simulation_amount)
                # определение имен файлов для таблиц и графиков
                csv_filename_delonge = filename_delonge + '.csv'
                xlsx_filename_delonge = filename_delonge + '.xlsx'
                png_filename_delonge = filename_delonge + '_' + distribution + '.png'
                png_hist_filename_delonge = filename_delonge + '_' + distribution + '_hist.png'

                get_agregated_results_delong(csv_filename_delonge,
                                             png_filename_delonge,
                                             png_hist_filename_delonge,
                                             distribution, z_score_list, count)
                # считываем csv и преобразуем его в xlsx
                df = pd.read_csv(csv_filename_delonge)
                df.to_excel(xlsx_filename_delonge, index=None, header=True)

        if experiment_parameters.do_delonge:
            files.delete_file(csv_filename_delonge)

        files.delete_file(csv_filepath_criteria)


def compute_roc(y_test, y_score):
    """
    Подсчет доли ложных и верных положительных классификаций.
    Подсчет площади под кривой.
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    return fpr, tpr, roc_auc, thresholds


def clf_result(clf):
    print(f"Коэффициент при КТИ: {clf.coef_[0][0]:.3f}\n"
          f"Коэффициент при неделях: {clf.coef_[0][1]:.3f}\n"
          f"Константа: {clf.intercept_[0]:.3f}")


def clf_indicators(y_test, y_predict):
    """
    Создание матрицы ошибок.
    Расчет на ее основе специфичности.
    Отдельным методом производится рассчет чувствительности.
    Специфичность - сколько здоровых было определено правильно из числа всех здоровых.
    Чувствительность - сколько больных было определено правильно из числа всех больных.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    specificity = tn / (tn + fp)
    recall = recall_score(y_test, y_predict)
    return specificity, recall


def confidence_interval(data, confidence=CONFIDENCE_LEVEL):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return dist.mean - h, dist.mean + h


def get_data_estimations(data):
    mean = np.mean(data)
    std = np.std(data)
    lower, upper = confidence_interval(data)
    return mean, std, lower, upper


def get_agregated_results_to_file(filepath, distr, sp, sn, roc):
    sp_mean, sp_std, sp_lower, sp_upper = get_data_estimations(sp)
    sn_mean, sn_std, sn_lower, sn_upper = get_data_estimations(sn)
    roc_mean, roc_std, roc_lower, roc_upper = get_data_estimations(roc)

    df = pd.DataFrame({'Distr': [distr],
                       'sp_mean': [sp_mean], 'sp_std': [sp_std],
                       'sn_mean': [sn_mean], 'sn_std': sn_std,
                       'roc_mean': [roc_mean], 'roc_std': [roc_std]})
    df = df.round(decimals=3)
    df.to_csv(filepath, index=None, header=not os.path.exists(filepath), mode='a')


def get_agregated_results_delong(filepath, plot_filepath, hist_filepath,
                                 distr, z, count):
    file = DELONG_PATH + '/Z-score_' + str(len(z)) + '_' + \
           filepath.split('_')[1][:-4] + \
           '_' + plot_filepath.split('_')[2][:-4] + '.dat'

    files.write_isw(file, z, "# " + plot_filepath.split('_')[2][:-4])

    edf = []
    # построение эмпирической функции распределения
    len_z = len(z)
    step = 1 / len_z
    for i in range(len_z):
        edf.append(step * i)
    edf = np.array(edf)

    # параметры стандартного нормального распределения
    params = [0, 1]
    # теоретические функции распределения и плотности
    cdf = st.norm.cdf(z, params[-2], params[-1])
    pdf = st.norm.pdf(z, params[-2], params[-1])

    # строим графики
    plots.plot_pdf_cdf_for_delong_test(z, edf, cdf, pdf, plot_filepath, hist_filepath)

    # формируем датафрейм и сохраняем
    df = pd.DataFrame({'Distr': [distr],
                       'Average Z': [np.mean(z)],
                       'Variance Z': [np.std(z)],
                       'N of Delonge p-val > 005': [float(count) / len_z * 100]})
    df = df.round(decimals=3)
    df.to_csv(filepath, index=None, header=not os.path.exists(filepath), mode='a')


if __name__ == "__main__":
    np.random.seed()
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    main_process()
