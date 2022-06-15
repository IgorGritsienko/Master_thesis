import warnings
import os

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score, roc_curve

from statistics import NormalDist

import scipy.stats as st

import files
import generation
import delong
import plots
import define_distribution



def main_process():
    """
    Основная функция программы.
    
    В input содержатся модели и кол-во записей.
    В params - число запусков эксперимента, флаг, обозначающей, будет ли тестовая
    выборка иметь другие параметры и соотношение тестовой и тренировочной выборки.
    frequency содержат частоты наблюдений.
    Остальные файлы создаются пустыми.
    """
    # для цикла 
    simulationAmount_list = [2000]
    sigma_norm_list =[0.008]#, 0.015, 0.026, 0.032, 0.04, 0.0455, 0.05, 0.083]
    sigma_anem_list =[0.083]#, 0.076, 0.065, 0.059, 0.051, 0.0455, 0.041, 0.008]
    distributions = ['uniform']#, 'laplace', 'norm', 'logistic', 'gumbel']
    # априорные вероятности
    prob_norm = 0.8
    prob_anem = 1 - prob_norm
    
    #prob_norm_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    input_ = 'input_data/input.txt'
    records = 'csv_files/records.csv'

    dif_test_records = 'csv_files/dif_test_records.csv'
    params_ = 'input_data/params.txt'
    frequency_norm = 'input_data/frequency_norm.txt'
    frequency_anem = 'input_data/frequency_anem.txt'
    
    delong_path = './DeLonge/'
    
    # если существует файл с записями, очистить его
    files.clear_file(records)
    files.clear_file(dif_test_records)
    files.create_dir('accuracy')
    files.create_dir('DeLonge')
    
    model_parameters = [float(i) for i in files.Read_parameters(input_)]

    normalAmount = int(model_parameters[3])
    anemiaAmount = int(model_parameters[7])

    #  считываем параметры генерации и присваиваем переменнным соответствующие значения
    parameters = files.Read_parameters(params_)

    #simulationAmount = int(parameters[0])                                     # количество запусков программы
    N_train_rate = float(parameters[2])                                        # доля обучающей выборки
    # количество всех записей
    N = normalAmount + anemiaAmount
    # создаем интервалы с вероятностными частотами
    intervals_norm = generation.Create_intervals(frequency_norm)
    intervals_anem = generation.Create_intervals(frequency_anem)

    # создали недели, на которых могут проводиться измерения
    # нормальные с 11 недели
    weeks_norm = range(11, 34)
    # анемия с 15 недели
    weeks_anem = range(15, 34)
    # делаем каждому элементу пару - будущее количество измероений на данной неделе
    weeks_norm = [[x] + [0] for x in weeks_norm]
    weeks_anem = [[x] + [0] for x in weeks_anem]

    #  создание объекта для стандартизации данных
    scaler = preprocessing.StandardScaler()
    
    for sigma_norm, sigma_anem in zip(sigma_norm_list, sigma_anem_list):
    #for prob_norm in prob_norm_list:
     #   prob_anem = 1 - prob_norm
        model_parameters[2] = sigma_norm
        model_parameters[6] = sigma_anem
        sigma_ratio = round(model_parameters[2] / model_parameters[6], 2)
        for simulationAmount in simulationAmount_list:
            for distribution in distributions:
                spec_list = []
                sens_list = []
                roc_list = []
                Z_score_list = []
                count = 0
                for i in range(simulationAmount):
                    Xy = generation.manipulate_gen_data(distribution, weeks_norm, weeks_anem, 
                                                   normalAmount, anemiaAmount,
                                                   scaler, records, intervals_norm, intervals_anem,
                                                   model_parameters)


                    #Xy = Xy.to_numpy()
                    X_train, X_test, y_train, y_test = train_test_split(Xy[:, :2], Xy[:, 2],
                                                                        test_size =  (1 - N_train_rate),
                                                                        random_state=np.random.randint(10000))
                    # позиция, с которой необходимо считывать записи для текущего эксперимента
                    # position = 0
                    # Xy, X_train, X_test, y_train, y_test, position = read_data_and_split(records,
                    #                                                                     position, N, (1 - N_train_rate))

                    lda = LinearDiscriminantAnalysis(solver='svd', priors=[prob_norm, prob_anem])
                    lda.fit(X_train, y_train)
                    plots.scatter_plot(X_test, y_test, lda.coef_[0][0], lda.coef_[0][1], lda.intercept_)


                    #lda1 = LinearDiscriminantAnalysis(solver = 'eigen', priors=[prob_norm, prob_anem])
                    #lda1.fit(X_train, y_train)
                    #plots.plot_lda_vs_qda(lda, qda, X_test, y_test)
                    
                    #вычисление специфичности и чувствительности
                    y_predict = lda.predict(X_test)
                    specificity, sensitivity = clf_indicators(y_test, y_predict)
                    
                    y_score = lda.decision_function(X_test)
                    fpr, tpr, roc_auc, thresholds = compute_ROC(y_test, y_score)
                    #plots.plot_ROC(fpr, tpr, roc_auc)
        
                    #получаем чувствительность и специфичность и ROC
                    #добавляем результаты в списки и повторяем процесс
                    spec_list.append(specificity)
                    sens_list.append(sensitivity)
                    roc_list.append(roc_auc)
                    
                np_roc_list = np.array(roc_list)
                roc_train_data, roc_valid_data, roc_test_data = define_distribution.split_criteria_data_into_sets(np_roc_list, simulationAmount)
        
                #Подбор наилучшего теоретического распределения для распределения данных ROC AUC
                data_type = 'roc'
                #define_distribution.check_for_agr(data_type, np_roc_list)
                
                dist = define_distribution.get_best_distribution(data_type, roc_train_data, roc_valid_data, roc_test_data, 
                                                                  sigma_ratio, simulationAmount, distribution)
        
                # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #     pd.set_option("colheader_justify", "left")
                #     print(f"Наилучшее распределение: {dist}")


# =============================================================================
#                 # сохранение данных после циклов по исследованию отклонения от нормальности
#                 filepath = 'accuracy/' + str(simulationAmount) + '_' + str(sigma_ratio) + '_' + str(prob_norm)
#                 csv_filepath = filepath + '.csv'
#                 xlsx_filepath = filepath + '.xlsx'
# 
#                 get_agregated_results_to_file(csv_filepath, distribution, spec_list, sens_list, roc_list)
#                 df = pd.read_csv(csv_filepath)
#                 df.to_excel(xlsx_filepath, index = None, header=True)
#             files.delete_file(csv_filepath)
# =============================================================================


# =============================================================================
#             # Блок для ДеЛонге, Вычисление Z-оценок, p-vaue, подбор теоретической
#             # функции нормального распределения для эмпирического распределения Z-оценок
#             #log10(p)  
#                     p_delong, Z_score = delong.delong_roc_test(y_test, 
#                                                                lda.predict_proba(X_test)[:, 1], 
#                                                                lda1.predict_proba(X_test)[:, 1]) 
#                     # для two-tailed нужно взять экспоненту
#                     #p_delong = pow(10, p_delong)
#                     Z_score_list.append(Z_score[0][0])
#                     if (p_delong > 0.05):
#                         count += 1
# 
#                 Z_score_list.sort()
# 
#                 # общее название файла для запуска
#                 filename = delong_path + str(simulationAmount) + '_' + str(sigma_ratio)
#                 # определение имен файлов для таблиц и графиков
#                 csv_filename = filename + '.csv'
#                 xlsx_filename = filename + '.xlsx'
#                 png_filename = filename + '_' + distribution + '.png'
#                 png_hist_filename = filename + '_' + distribution + '_hist.png'
#                 
#                 get_agregated_results_to_file_delong(csv_filename, png_filename, png_hist_filename, 
#                                                      distribution, Z_score_list, count)
#                 # считываем csv и преобразуем его в xlsx
#                 df = pd.read_csv(csv_filename)
#                 df.to_excel(xlsx_filename, index = None, header=True)
#             files.delete_file(csv_filename)
#                 
# =============================================================================





def compute_ROC(Y_test, Y_score):
    """
    Подсчет доли ложных и верных положительных классификаций.
    Подсчет площади под кривой.
    """

    fpr, tpr, thresholds = roc_curve(Y_test, Y_score, pos_label=1)
    roc_auc = roc_auc_score(Y_test, Y_score)
    return fpr, tpr, roc_auc, thresholds


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

def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return dist.mean - h, dist.mean + h

def get_estimations_on_data(data):
    mean = np.mean(data)
    std = np.std(data)
    lower, upper = confidence_interval(data)
    return mean, std, lower, upper

def get_agregated_results_to_file(filepath, distr, sp, sn, roc):
    
    sp_mean, sp_std, sp_lower, sp_upper = get_estimations_on_data(sp)
    sn_mean, sn_std, sn_lower, sn_upper = get_estimations_on_data(sn)
    roc_mean, roc_std, roc_lower, roc_upper = get_estimations_on_data(roc)

    df = pd.DataFrame()
    df["Distr"] = [distr]
    
    df["sp_mean"] = [sp_mean]
    df["sp_std"] = [sp_std]
    
    df["sn_mean"] = [sn_mean]
    df["sn_std"] = [sn_std]

    df["roc_mean"] = [roc_mean]
    df["roc_std"] = [roc_std]

    df = df.round(decimals=3)
    df.to_csv(filepath, index = None, header=not os.path.exists(filepath), mode='a')

def get_agregated_results_to_file_delong(filepath, plot_filepath, hist_filepath,
                                         distr, Z, count):
    params = [0, 1]
    
    file = './DeLonge/Z-score_' + str(len(Z)) + '_' + filepath.split('_')[1][:-4] + '_' + plot_filepath.split('_')[2][:-4] + '.dat'
    files.write_to_txt_ISW(file, Z, "# " + plot_filepath.split('_')[2][:-4])

    edf = []
    #построение эмпирической функции распределения
    len_Z = len(Z)
    step = 1 / len_Z
    for i in range(len_Z):
        edf.append(step * i)
    edf = np.array(edf)

    # теоретические функции распределения и плотности
    cdf = st.norm.cdf(Z, params[-2], params[-1])
    pdf = st.norm.pdf(Z, params[-2], params[-1])
    
    # строим графики
    plots.plot_pdf_cdf_for_delong_test(Z, edf, cdf, pdf, params, plot_filepath, hist_filepath)

    # формируем датафрейм и сохраняем
    df = pd.DataFrame()
    df["Distr"] = [distr]
    df["Average Z"] = [np.mean(Z)]
    df["Variance Z"] = [np.std(Z)]
    df["N of DeLonge p-val > 005"] = [float(count) / len_Z * 100]
    df = df.round(decimals=3)
    df.to_csv(filepath, index = None, header=not os.path.exists(filepath), mode='a')

def get_effective_quantile(dataset, distribution, quantile):
    dist_quantile = distribution.ppf(quantile)
    effective_quantile = sum(dataset <= dist_quantile) / len(dataset)
    return(effective_quantile)

if __name__ == "__main__":
    # Начало программы
    np.random.seed()  
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    #plots.plot_errors_dist()
    main_process()

