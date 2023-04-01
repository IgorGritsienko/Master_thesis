import numpy as np
import pandas as pd
import scipy.stats as st
from numpy.random import default_rng

import files
import plots

DISTRIBUTIONS_NUM_TO_PLOT = 5


def split_criteria_data_into_sets(criteria_data, n):
    """
    На входе имеем список результатов по одному из критериев и размер массива (кол-во запусков эксперимента)
    Разбиваем его на тренировочную, валидационную и тестовую подвыборку.
    Соотношение 50-30-20.
    """

    train_rate = 0.5
    validate_rate = 0.3

    # перемешать данные
    rng = default_rng()
    rng.shuffle(criteria_data)

    train_data = criteria_data[: int(n * train_rate)]
    validate_data = criteria_data[int(n * train_rate): int(n * (train_rate + validate_rate))]
    test_data = criteria_data[int(n * (train_rate + validate_rate)):]
    return train_data, validate_data, test_data


def kolm_dist(sorted_test_data, dist, params):
    # теоретическая функция распределения по значениям иксов
    cdf = dist.cdf(sorted_test_data, *params[0][:-2], loc=params[0][-2], scale=params[0][-1])
    # для эмпирической функции распределения в качестве начального значения берем
    # значение теоретической функции распределения в первой точке
    edf = []
    # построение эмпирической функции распределения
    step = 1 / sorted_test_data.size
    for i in range(sorted_test_data.size):
        edf.append(step * i)
    edf = np.array(edf)
    # считаем расстояние Колмогорова
    kolmogorov_distance = abs(edf - cdf).max()
    return edf, kolmogorov_distance


def get_best_distribution(dir_path, distribution, roc_list, sigma_ratio, simulation_amount):
    # создаем директорию для конкретного случая
    dir_name = dir_path + '/' + str(distribution) + "/" + str(sigma_ratio) + "_" + str(simulation_amount)
    files.create_dir(dir_name)

    np_roc_list = np.array(roc_list)
    train_data, validate_data, test_data = split_criteria_data_into_sets(np_roc_list, simulation_amount)

    # Set up empty lists to store results
    d_values = []
    p_values = []
    # parameters = []
    significance_level = 0.05

    # dist_names = ["alpha","anglit","arcsine","beta","betaprime","bradford","burr","cauchy","chi","chi2","cosine",
    #     "dgamma","dweibull","erlang","expon","exponnorm","exponweib","exponpow","f","fatiguelife","fisk",
    #     "foldcauchy","foldnorm","genlogistic","genpareto","gennorm","genexpon",
    #     "genextreme","gausshyper","gamma","gengamma","genhalflogistic","gilbrat","gompertz","gumbel_r",
    #     "gumbel_l","halfcauchy","halflogistic","halfnorm","halfgennorm","hypsecant","invgamma","invgauss",
    #     "invweibull","johnsonsb","johnsonsu","ksone","kstwobign","laplace","levy","levy_l","levy_stable",
    #     "logistic","loggamma","loglaplace","lognorm","lomax","maxwell","mielke","nakagami","ncx2","ncf",
    #     "nct","norm","pareto","pearson3","powerlaw","powerlognorm","powernorm","rdist","reciprocal",
    #     "rayleigh","rice","recipinvgauss","semicircular","t","triang","truncexpon","truncnorm","tukeylambda",
    #     "uniform","vonmises","vonmises_line","wald","weibull_min","weibull_max"]

    # для ROC AUC

    dist_names = ["alpha", "betaprime", "burr", "dgamma",
                  "dweibull", "exponnorm", "f", "fatiguelife", "fisk",
                  "foldnorm", "genlogistic", "gennorm", "genextreme",
                  "gumbel_l", "hypsecant", "invgamma", "johnsonsb", "johnsonsu", "laplace", "levy_stable",
                  "logistic", "loggamma", "lognorm", "mielke", "nakagami", "norm", "pearson3",
                  "powerlognorm", "powernorm", "recipinvgauss", "t", "tukeylambda", "weibull_min"]

    # подбор параметров для распределений
    # стадия обучения
    params_list = []
    # _distn_names
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(train_data)

        # Прошедшие согласие распределения заносятся в новый массив
        # стадия валидации
        d, p = st.kstest(validate_data, dist_name, args=param)
        p = np.around(p, 3)
        d_values.append(d)
        p_values.append(p)
        params_list.append(param)

    # создаем таблицу (датафрейм) и добавляем столбцы с данными
    results = pd.DataFrame({'Distr': dist_names, 'D_stats': d_values, 'p_value': p_values, 'Parameters': params_list})
    # сортируем по p-value
    results.sort_values(['p_value'], inplace=True, ascending=False)
    # сохраняем в файл датафрейм
    results.to_csv(dir_name + '/agr_crit.csv', sep=';', index=False, mode='a', header=True, float_format='%.6f')
    # удаляем распределения, не прошедшие контроль p-value
    results.drop(results[results['p_value'] < significance_level].index, inplace=True)
    # сохраняем в файл датафрейм
    results.to_csv(dir_name + '/passed_crit.csv', sep=';', index=False, mode='a', header=True, float_format='%.6f')

    # results.sort_values(['D_stats'], inplace=True)
    sorted_test_data = np.sort(test_data, kind='heapsort')
    kolmogorov_distance_list = []
    edf_list = []
    for dist_name in results['Distr'].values:
        dist = getattr(st, dist_name)
        # находим строку в датафрейме с нужным распределением
        row = results[results['Distr'] == dist_name]
        # берем столбец с параметрами (мю, сигма) для дальнейшего использования
        params = (row.loc[:, 'Parameters']).values
        # вычисляем эмпирическую ф-ю распределения и расстояние колмогорова
        edf, kolmogorov_distance = kolm_dist(sorted_test_data, dist, params)
        # добавляем в список для таких же величин, но при других распределениях
        kolmogorov_distance_list.append(kolmogorov_distance)
        edf_list.append(edf)

    # добавляем в датафрейм новый столбец
    results['Kolm Dist'] = kolmogorov_distance_list
    results['edf'] = edf_list

    # сортируем по возрастанию расстояния Колмогорова
    results.sort_values(['Kolm Dist'], inplace=True)
    results.to_csv(dir_name + '/kolm_dist.csv', sep=';', index=False, mode='a', header=True, float_format='%.6f')

    # берем первое значение по индексу
    # если таковых нет, выходим из функции
    if results.size == 0:
        return
    # получаем строковое значение названия распределения
    best_dist = results.iloc[0]

    # подсчитываем кол-во строк
    size = results['Distr'].values.size
    if size > DISTRIBUTIONS_NUM_TO_PLOT:
        size = DISTRIBUTIONS_NUM_TO_PLOT
    # строим графики теоретической и эмпирической функций
    for i in range(size):
        dist = getattr(st, results['Distr'].values[i])
        params = results['Parameters'].values[i]
        cdf = dist.cdf(sorted_test_data, *params[:-2], loc=params[-2], scale=params[-1])
        edf = results['edf'].values[i]
        plots.plot_cdf_edf(results['Distr'].values[i], sorted_test_data, cdf, edf, dir_name)

    return best_dist
