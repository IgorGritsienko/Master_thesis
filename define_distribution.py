import pandas as pd
import numpy as np

import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names

import files
import plots

def get_best_distribution(data_type, train_data, validate_data, test_data, sigma_ratio, simulationAmount):
    # создаем директорию для конкретного случая
    dir_name = "output/" + data_type + "_ratio_" + str(sigma_ratio) + "_amount_" + str(simulationAmount)
    files.create_dir(dir_name)
    
    # Set up empty lists to stroe results
    D_values = []
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
    # dist_names = ["alpha","betaprime","burr","dgamma",
    #     "dweibull","exponnorm","f","fatiguelife","fisk",
    #     "foldnorm","genlogistic","gennorm", "genextreme",
    #     "gumbel_l","hypsecant","invgamma","johnsonsb","johnsonsu","laplace","levy_stable",
    #     "logistic","loggamma","lognorm","mielke","nakagami","norm","pearson3",
    #     "powerlognorm","powernorm","recipinvgauss","t","tukeylambda","weibull_min"]


    # для ROC AUC на 1700 запусках: 50-30-20
    # в порядке убываничя p-value
    #dist_names = ["exponweib", "gumbel_l", "exponpow", "johnsonsb", "pearson3",
    #              "beta", "johnsonsu", "gengamma", "powernorm"]
    
    # для ROC AUC на 2000 запусках: 50-30-20
    # ДЛЯ ПРЕЗЕНТАЦИИ ГРАФИКИ
    dist_names = ["powernorm",]#"exponweib", "gumbel_l", "johnsonsb", "pearson3", "johnsonsu"]
    
        #_distn_names
        # подбор параметров для распределений
        # стадия обучения
    params_list = []
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(train_data)

        # Obtain the KS test P statistic, round it to 3 decimal places
        # Прошедшие согласие распределения заносятся в новый массив
        # стадия валидации
        D, p = st.kstest(validate_data, dist_name, args=param)
        p = np.around(p, 3)
        D_values.append(D)
        p_values.append(p)
        params_list.append(param)
    
    # создаем таблицу (датафрейм) и добавляем столбцы с данными
    results = pd.DataFrame()
    results['Distr'] = dist_names
    results['D_stats'] = D_values
    results['p_value'] = p_values
    results['Parameters'] = params_list
    # сортируем по p-value
    results.sort_values(['p_value'], inplace=True, ascending=False)
    # сохраняем в файл датафрейм
    results.to_csv(dir_name + '/agr_crit.csv', sep=';',index=False, mode='a', header=True, float_format='%.6f')
    # удаляем распределения, не прошедшие контроль p-value
    results.drop(results[results['p_value'] < significance_level].index, inplace=True)
    # сохраняем в файл датафрейм
    results.to_csv(dir_name + '/passed_crit.csv', sep=';',index=False, mode='a', header=True, float_format='%.6f')

    #results.sort_values(['D_stats'], inplace=True)
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
    results.to_csv(dir_name + '/kolm_dist.csv', sep=';',index=False, mode='a', header=True, float_format='%.6f')

    # сортируем по возрастанию расстояния Колмогорова
    results.sort_values(['Kolm Dist'], inplace=True)
    # берем первое значение по индексу
    # если таковых нет, выходим из функции
    if (results.size == 0):
        return
    # получаем строковое значение названия распределения
    best_dist = results.iloc[0]
    # по строковому значению получаем объект-распределение
    dist = getattr(st, best_dist['Distr'])
    params = best_dist['Parameters']
    edf = best_dist['edf']
    # вычисляем значения для теоритической ф-и распределения
    cdf = dist.cdf(sorted_test_data, *params[:-2], loc=params[-2], scale=params[-1])
    # отрисовываем теоретическую и эмпирическую функции
    plots.plot_cdf_edf(dist_name, sorted_test_data, cdf, edf, dir_name)
    
    # сортируем по убыванию p-value
    results.sort_values(['p_value'], inplace=True, ascending=False)
    # подсчитываем кол-во строк
    size = results['Distr'].values.size
    # если эл-ов меньше 5, то берем кол-во строк
    if size > 6:
        size = 6
    # строим графики теоретической и эмпирической функций
    for i in range(size):
        dist = getattr(st, results['Distr'].values[i])
        params = results['Parameters'].values[i]
        cdf = dist.cdf(sorted_test_data, *params[:-2], loc=params[-2], scale=params[-1])
        edf = results['edf'].values[i]
        plots.plot_cdf_edf(results['Distr'].values[i], sorted_test_data, cdf, edf, dir_name)
    
    return best_dist

    # веса для столбцов гистограммы - одинаковые 
    #weights=np.ones_like(test_data) / len(test_data)
    #ax.hist(test_data, weights=weights)


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


def check_for_agr(data_type, data):
    dir_name = "output/" + data_type + "_ratio_1.0_amount_2000"
    file_name = dir_name + '/kolm_dist.csv'
    df = pd.read_csv(file_name, delimiter=';')
    for i in range(len(df.index)):
        dist_name = df.iloc[i, 0]
        params = df.iloc[i, 3]
        # преобразовать строку с параметрами в кортеж
        params = tuple(map(float, params[1 : -1 : ].split(', ')))
        D, p = st.kstest(data, dist_name, args=params)
        print(f"dist: {dist_name}\nparams: {params}\np-value: {p}")
