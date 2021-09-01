import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import math
import statistics

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import scipy.stats as st

import generation
import files
import plots
import define_distribution

def Tune_Hyperparameters(model, X, y):
    """
    Подбор гиперпараметров.
    Создается сетка с различными параметрами и происходит перебор
    каждый с каждым.
    Происходит сравнение по критерию.
    На данный момент - по точности.
    Наилучший вариант может быть использован в дальнейшем.
    На данный момент возвращается только решатель (в качестве примера), но
    при переборе используются еще и различные априорные вероятности.
    """

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = dict()
    grid['solver'] = ['svd', 'lsqr', 'eigen']
    grid['priors'] = [[z / 10, (10 - z) / 10] for z in range(1, 10)]
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    # print('Mean Accuracy: %.3f' % results.best_score_)
    # print('Config: %s' % results.best_params_)
    return results.best_params_['solver']


def compute_ROC_curve_ROC_area(Y_test, Y_score):
    """
    Подсчет доли ложных и верных положительных классификаций.
    Подсчет площади под кривой.
    """

    fpr, tpr, _ = roc_curve(Y_test, Y_score)
    roc_auc = roc_auc_score(Y_test, Y_score)
    return fpr, tpr, roc_auc


def clf_results(clf):
    print(f"Коэффициент при КТИ: {clf.coef_[0][0]:.3f}\nКоэффициент при неделях: {clf.coef_[0][1]:.3f}\nКонстанта: {clf.intercept_[0]:.3f}")


def clf_indicators(y_test, y_predict):
    """
    Создание матрицы ошибок.
    Рассчет на ее основе специфичности.
    Отдельным методом производится рассчет чувствительности.
    Специфичность - сколько здоровых было определено правильно из числа всех здоровых.
    Чувствительность - сколько больных было определено правильно из числа всех больных.
    """

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    specificity = tn / (tn + fp)
    recall = recall_score(y_test, y_predict)
    return specificity, recall


def read_data_and_split(filename, position, shift, test_amount):
    """
    Считываем shift записей из файла filename, начиная с позиции position.
    Каждый раз роисходит сдвиг считывания позиции.
    Считанные данные разбиваются на тренировочные и тестовые выборки.
    """

    Xy = files.Read_csv(filename, shift, position)
    position += shift
    Xy = Xy.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(Xy[:, :2], Xy[:, 2], test_size = test_amount, random_state=np.random.randint(10000))
    return Xy, X_train, X_test, y_train, y_test, position


def main_process():
    """
    Основная функция программы.
    
    В input содержатся модели и кол-во записей.
    В params - число запусков эксперимента, флаг, обозначающей, будет ли тестовая
    выборка иметь другие параметры и соотношение тестовой и тренировочной выборки.
    frequency содержат частоты наблюдений.
    Остальные файлы создаются пустыми.
    """

    input_ = 'input_data/input.txt'
    records = 'csv_files/records.csv'
    dif_test_records = 'csv_files/dif_test_records.csv'
    train_sample = 'csv_files/train_sample.csv'
    test_sample = 'csv_files/test_sample.csv'
    params_ = 'input_data/params.txt'
    frequency_norm = 'input_data/frequency_norm.txt'
    frequency_anem = 'input_data/frequency_anem.txt'

    # если существует файл с записями, очистить его
    files.clear_file(records)
    files.clear_file(dif_test_records)

    #  считываем данные модели и присваиваем переменнным соответствующие значения
    model_parameters = files.Read_parameters(input_)
    normalAmount = int(model_parameters[3])
    anemiaAmount = int(model_parameters[7])

    #  считываем параметры генерации и присваиваем переменнным соответствующие значения
    parameters = files.Read_parameters(params_)

    simulationAmount = int(parameters[0])                                      # количество запусков программы
    isTestSampleCustom = int(parameters[1])                                    # будет ли отдельная тестовая выборка со своими параметрами
    N_train_rate = float(parameters[2])                                        # доля обучающей выборки

    # размер обучающей выборки
    N_train_records = int((normalAmount + anemiaAmount) * N_train_rate)
    # количество всех записей
    N = normalAmount + anemiaAmount

    # объем записей для теста (в случае использования тест. выборки с другими параметрами)
    test_normal_amount = round(normalAmount * (1 - N_train_rate))
    test_anemia_amount = round(anemiaAmount * (1 - N_train_rate))

    # создаем интервалы с вероятностными частотами
    intervals_norm = generation.Create_intervals(frequency_norm)
    intervals_anem = generation.Create_intervals(frequency_anem)

    # анемия с 15 недели
    # нормальные с 11 недели
    weeks_norm = range(11, 34)                                                 # создали недели, на которых могут проводиться измерения
    weeks_norm = [[x] + [0] for x in weeks_norm]                               # делаем каждому элементу пару - будущее количество измероений на данной неделе
    weeks_anem = range(15, 34)
    weeks_anem = [[x] + [0] for x in weeks_anem]

    spec_list = []
    sens_list = []
    roc_list = []
    #  создание объекта для стандартизации данных
    scaler = preprocessing.StandardScaler()

    for i in range(simulationAmount):
        generation.manipulate_gen_data(weeks_norm, weeks_anem, normalAmount, anemiaAmount,
                                       scaler, records, intervals_norm, intervals_anem, model_parameters)
        #  создание файла в случае инородной тестовой выборки
        if (isTestSampleCustom == 1):
            # моделирование тестовой выборки с другими параметрами
            generation.manipulate_gen_data(weeks_norm, weeks_anem, scaler, test_normal_amount, test_anemia_amount,
                                           scaler, dif_test_records, intervals_norm, intervals_anem, model_parameters)

    #  изначальная априорная вероятность для работы дискриминантного анализа
    prob_norm = 0.8
    prob_anem = 1 - prob_norm
    #while (prob_norm <= 0.95):

    #  позиция, с которой необходимо считывать записи для текущего эксперимента.
    position = 0
    position_dif_test = 0

    #  каждый раз позиция увеличивается на N
    position_shift_dif_test = test_normal_amount + test_anemia_amount

    for i in range(simulationAmount):
        Xy, X_train, X_test, y_train, y_test, position = read_data_and_split(records, position, N, (1 - N_train_rate))
        # если для теста используется другая выборка
        if (isTestSampleCustom == 1):
            _, _, X_test, _, y_test, position_dif_test = read_data_and_split(dif_test_records, position_dif_test,
                                                                          position_shift_dif_test, int(N * (1 - N_train_rate)))

        clf = LinearDiscriminantAnalysis()
        # перебор классификаторов и параметра shrinkage. критерий выбора - наилучшая точность
        # используется отдельно, вне основной процедуры, для выбора наилучших гиперпараметров
        solver = Tune_Hyperparameters(clf, Xy[:, :2], Xy[:, 2])

        clf = LinearDiscriminantAnalysis(solver = solver, priors=[prob_norm, prob_anem])
        # тренируем
        clf.fit(X_train, y_train)
        # предсказываем
        y_predict = clf.predict(X_test)
        # clf_results(clf)
        #  график рассеивания
        # plots.scatter_plot(X_test, y_test, clf.coef_[0][0], clf.coef_[0][1], clf.intercept_)

        #  получаем чувствителньость и специфичность
        specificity, sensitivity = clf_indicators(y_test, y_predict)

        y_score = clf.decision_function(X_test)
        fpr, tpr, roc_auc = compute_ROC_curve_ROC_area(y_test, y_score)
        # plots.plot_ROC(fpr, tpr, roc_auc)

# =============================================================================
#         сохранение данных, на которых проводится тренировка и тест
#         y_train = np.reshape(y_train, [np.size(y_train), 1])
#         Save_sample_to_csv(np.hstack((X_train, y_train)), train_sample, _mode = 'w')
#         y_test = np.reshape(y_test, [np.size(y_test), 1])
#         Save_sample_to_csv(np.hstack((X_test, y_test)), test_sample, _mode = 'w')
#         coefs = np.reshape(np.array([clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0]]), [-1, 3])
# =============================================================================

        #  добавляем результаты в списки и повторяем процесс
        spec_list.append(specificity)
        sens_list.append(sensitivity)
        roc_list.append(roc_auc)


    np_spec_list = np.array(spec_list)
    np_sens_list = np.array(sens_list)
    np_roc_list = np.array(roc_list)

    files.quality_criteria_results_print_save_to_file(np_spec_list, np_sens_list, np_roc_list, prob_norm, prob_anem)
    plots.plot_criteria_distributions(np_spec_list, np_sens_list, np_roc_list)

    #prob_norm = prob_norm + 0.1 # для цикла while при переборе априорных вероятностей

    #np.random.shuffle(np_spec_list) #DevSkim: ignore DS148264
    length = int(np_spec_list.size / 2)
    np_spec_list_train, np_spec_list_test = np_spec_list[:length], np_spec_list[length:]

    # количество распределений для отрисовки
    number_distributions_to_plot = 1
    dist, params = define_distribution.get_best_distribution(np_spec_list_train, number_distributions_to_plot)
    D, p = st.kstest(np_spec_list_test, dist, args=params)
    print(dist, D, p)
    


if __name__ == "__main__":
    # Начало программы
    #  для фиксации значений выставляем seed
    np.random.seed(0)
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    main_process()
