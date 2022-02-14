import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng
from numpy.random import randint


from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import plots

import plots
import delong
import plots
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
def split_criteria_data_into_sets(criteria_data, N):
    """
    На входе имеем список результатов по одному из критериев и размер массива (кол-во запусков эксперимента)
    Разбиваем его на тренировочную, валидационную и тестовую подвыборку.
    Соотношение 70-15-15.
    """
    
    train_rate = 0.5
    validate_rate = 0.3
    #train_rate = 0.7
    #validate_rate = 0.15
    
    # перемешать данные
    rng = default_rng()
    rng.shuffle(criteria_data)
    
    train_data = criteria_data[ : int(N * train_rate)]
    validate_data = criteria_data[int(N * train_rate) : int(N * (train_rate + validate_rate))]
    test_data = criteria_data[int(N * (train_rate + validate_rate)) : ]
    return train_data, validate_data, test_data

def calc_data_for_compbdt(array1, array2):
    if array1[0][0] > array2[0][0]:
        s11 = array2[0][0]
    else:
        s11 = array1[0][0]
       
    if array1[0][1] > array2[0][1]:        
        s10 = 0
    else:
        s10 = array2[0][1] - array1[0][1]
        
    if array1[0][1] > array2[0][1]:        
        s01 = array1[0][1] - array2[0][1]
    else:
        s01 = 0
        
    if array1[0][1] > array2[0][1]:        
        s00 = array2[0][1]
    else:
        s00 = array1[0][1] 
        
    if array1[1][0] > array2[1][0]:        
        r11 = array2[1][0]
    else:
        r11 = array1[1][0] 
    
    if array1[1][0] > array2[1][0]:
        r10 = array1[1][0] - array2[1][0]
    else:
        r10 = 0
    
    if array1[1][0] > array2[1][0]:
        r01 = 0
    else:
        r01 = array2[1][0] - array1[1][0]
        
    if (array1[1][1] > array2[1][1]):
        r00 = array2[1][1]
    else:
        r00 = array1[1][1]
        
    return s11, s10, s01, s00, r11, r10, r01, r00

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
    compbdt_old = './compbdt.csv'
    dif_test_records = 'csv_files/dif_test_records.csv'
    #train_sample = 'csv_files/train_sample.csv'
    #test_sample = 'csv_files/test_sample.csv'
    params_ = 'input_data/params.txt'
    frequency_norm = 'input_data/frequency_norm.txt'
    frequency_anem = 'input_data/frequency_anem.txt'
    
    compbdt_data_file = "./compbdt.csv"
    compbdt_prog_path = 'C:\\Users/Igor/Documents/compbdt.R'
    
    # если существует файл с записями, очистить его
    files.clear_file(records)
    files.clear_file(dif_test_records)
    model_parameters = [float(i) for i in files.Read_parameters(input_)]
    
    # запоминаем отношение сигм для дальнейшего использования
    sigma_ratio = round(model_parameters[2] / model_parameters[6], 2)
    
    
    model_parameters = files.Read_parameters(input_)
    model_parameters = files.Read_parameters(input_)
    model_parameters = files.Read_parameters(input_)
    normalAmount = int(model_parameters[3])
    anemiaAmount = int(model_parameters[7])

    #  считываем параметры генерации и присваиваем переменнным соответствующие значения
    parameters = files.Read_parameters(params_)

    simulationAmount = int(parameters[0])                                      # количество запусков программы
    isTestSampleCustom = int(parameters[1])                                    # будет ли отдельная тестовая выборка со своими параметрами
    N_train_rate = float(parameters[2])                                        # доля обучающей выборки

    # размер обучающей выборки
    # N_train_records = int((normalAmount + anemiaAmount) * N_train_rate)
    # количество всех записей
    N = normalAmount + anemiaAmount

    # объем записей для теста (в случае использования тест. выборки с другими параметрами)
    test_normal_amount = round(normalAmount * (1 - N_train_rate))
    test_anemia_amount = round(anemiaAmount * (1 - N_train_rate))

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

    spec_list = []
    sens_list = []
    roc_list = []
    #  создание объекта для стандартизации данных
    scaler = preprocessing.StandardScaler()

    for i in range(simulationAmount):
        # "коридор дисперсий"
        #sigma_deviation = randint(-455, 546) * 0.0001
        # к дисперсиям обоих классов добавляем разброс
        #model_parameters[2] += sigma_deviation
       # model_parameters[6] += sigma_deviation
        
        generation.manipulate_gen_data(weeks_norm, weeks_anem, normalAmount, anemiaAmount,
                                       scaler, records, intervals_norm, intervals_anem, model_parameters)
        #  создание файла в случае инородной тестовой выборки
        if (isTestSampleCustom == 1):
            # моделирование тестовой выборки с другими параметрами
            generation.manipulate_gen_data(weeks_norm, weeks_anem, test_normal_amount, test_anemia_amount,
                                           scaler, dif_test_records, intervals_norm, intervals_anem, model_parameters)

    # априорные вероятности
    prob_norm = 0.8
    prob_anem = 1 - prob_norm

    # позиция, с которой необходимо считывать записи для текущего эксперимента.
    # каждый раз позиция увеличивается на N (при инородной тестовой выборке)
    #  каждый раз позиция увеличивается на N
    #  каждый раз позиция увеличивается на N
    #  каждый раз позиция увеличивается на N
    position_shift_dif_test = test_normal_amount + test_anemia_amount

    # список для хранения результатов Z оценок
    Z_score_list = []
    p_list = []
    count = 0
    
    compbdt_data_list = []
    i = 1
    while i < simulationAmount:
        Xy, X_train, X_test, y_train, y_test, position = read_data_and_split(records, position, N, (1 - N_train_rate))
        print(f"success: {i}")
        # если для теста используется другая выборка
        if (isTestSampleCustom == 1):
            _, _, X_test, _, y_test, position_dif_test = read_data_and_split(dif_test_records, position_dif_test,
                                                                          position_shift_dif_test, int(N * (1 - N_train_rate)))
        #solver = Tune_Hyperparameters(clf, Xy[:, :2], Xy[:, 2])
      
        lda = LinearDiscriminantAnalysis(solver='svd', priors=[prob_norm, prob_anem])
        lda.fit(X_train, y_train)
        # plots.scatter_plot(X_test, y_test, lda.coef_[0][0], lda.coef_[0][1], lda.intercept_)


        qda = QuadraticDiscriminantAnalysis(priors=[prob_norm, prob_anem])
        #qda = LinearDiscriminantAnalysis(solver='lsqr', priors=[prob_norm, prob_anem])
        qda.fit(X_train, y_train)
        
        # plots.plot_lda_vs_qda(lda, qda, X_test, y_test)
        # plots.scatter_plot(X_test, y_test, clf.coef_[0][0], clf.coef_[0][1], clf.intercept_)
        y_predict_lda = lda.predict(X_test)
        # матрица ошибок sklearn (инверсирована)
        cf = confusion_matrix(y_test, y_predict_lda)
        # матрица ошибок (стандартная)
        cf_lda = [[cf[1][1], cf[1][0]], [cf[0][1], cf[0][0]]]
        # print(f"lda\n{cf_lda[0]}\n{cf_lda[1]}")
        # print(cf)
        y_predict_qda = qda.predict(X_test)
        cf = confusion_matrix(y_test, y_predict_qda)
        cf_qda = [[cf[1][1], cf[1][0]], [cf[0][1], cf[0][0]]]
        # print(f"qda\n{cf_qda[0]}\n{cf_qda[1]}")
        
        # вычисление специфичности и чувствительности
        specificity_lda, sensitivity_lda = clf_indicators(y_test, y_predict_lda)
        specificity_qda, sensitivity_qda = clf_indicators(y_test, y_predict_qda)
        # print(round(sensitivity_lda, 5), round(specificity_lda, 5))
        # print(round(sensitivity_qda, 5), round(specificity_qda, 5))
        



# =============================================================================
#         y_score = lda.decision_function(X_test)
#         fpr, tpr, roc_auc = compute_ROC_curve_ROC_area(y_test, y_score)
#         plots.plot_ROC(fpr, tpr, roc_auc)
#         
#           # получаем чувствительность и специфичность и ROC
#         добавляем результаты в списки и повторяем процесс
        # spec_list.append(specificity)
        # sens_list.append(sensitivity)
        
        compbdt_data = [s11, s10, s01, s00, r11, r10, r01, r00] = calc_data_for_compbdt(cf_lda, cf_qda)
        if (s10 + s01 == 0 or r10 + r01 == 0):
            i -= 1
            generation.manipulate_gen_data(weeks_norm, weeks_anem, normalAmount, anemiaAmount,
                                       scaler, records, intervals_norm, intervals_anem, model_parameters)
            continue
        i += 1
        compbdt_data_list.append(compbdt_data)
    # Создаем файл для сохранения данных для программы compbdt
    np_compbdt_data_file = np.array(compbdt_data_list)
    files.save_tests_results_to_csv(np_compbdt_data_file, compbdt_data_file)
    
    # Импортируем пакет и вызываем скрипт на языке R
    print(np_compbdt_data_file.shape)
    print(simulationAmount)
    import rpy2.robjects as robjects
    robjects.r.source(compbdt_prog_path, encoding="utf-8")
    
    # полученные значения специфичности, чувствительности и рок аук преобразуем в numpy array
    # np_spec_list = np.array(spec_list)
    # np_sens_list = np.array(sens_list)
    # np_roc_list = np.array(roc_list)
    

    plots.plot_criteria_distributions(np_spec_list, np_sens_list, np_roc_list)
    plots.plot_criteria_distributions(np_spec_list, np_sens_list, np_roc_list)
    plots.plot_criteria_distributions(np_spec_list, np_sens_list, np_roc_list)

# =============================================================================
#         # Блок для ДеЛонге, Вычисление Z-оценок, p-vaue, подбор теоретической
#         # функции нормального распределения для эмпирического распределения Z-оценок
#         #log10(p)
#         p_delong, Z_score = delong.delong_roc_test(y_test, lda.predict_proba(X_test)[:, 1], qda.predict_proba(X_test)[:, 1]) 
#         # для two-tailed нужно взять экспоненту
#         # p_delong = pow(10, p_delong)
#         p_list.append(p_delong)
#         Z_score_list.append(Z_score[0][0])
#         # print(f"Z_score for DeLong's test is: {np.ravel(Z_score)}")
#         # print(f"p_value for DeLong's test is: {np.ravel(p_delong)}")
#         if (p_delong > 0.05):
#             count += 1
#         
# 
#         
#     Z_score_list.sort()
#     params = scipy.stats.norm.fit(Z_score_list)
#     #Z_score_list = preprocessing.scale(Z_score_list)
#     edf = []
#     #построение эмпирической функции распределения
#     step = 1 / len(Z_score_list)
#     for i in range(len(Z_score_list)):
#         edf.append(step * i)
#     edf = np.array(edf)
#     
#     cdf = scipy.stats.norm.cdf(Z_score_list, params[-2], params[-1])
#     
#     fig, ax = plt.subplots()
#     ax.plot(Z_score_list, cdf,
#             'r-', lw=3, alpha=0.6, label='Теоретическая')
#     ax.plot(Z_score_list, edf,
#             'b--', lw=3, alpha=0.6, label='Эмпирическая')
#     
#     filename = 'Z-score_edf'
#     plt.savefig(filename)
#     plt.show()
#     
#     print(f"DeLong's p-value > 0.05: {round(float(count) / simulationAmount, 2) * 100}%")
#     print(f"mu: {params[-2]}\nsigma: {params[-1]}")
#     D, p = scipy.stats.kstest(Z_score_list, 'norm', args=params)
#     print(f"D: {D}\np-value: {p}")
#     
#     files.write_to_txt('p-value.txt', p_list)
#     files.write_to_txt('Z_score.txt', Z_score_list)
    
    # сохраняем результаты в файл
# =============================================================================
#     #files.quality_criteria_results_print_save_to_file(np_spec_list, np_sens_list, np_roc_list, prob_norm, prob_anem,
#     #                                                  sigma_ratio, simulationAmount)
#     #plots.plot_criteria_distributions(np_spec_list, np_sens_list, np_roc_list)
# 
#     # roc_train_data, roc_valid_data, roc_test_data = split_criteria_data_into_sets(np_roc_list, simulationAmount)

#     # Подбор наилучшего теоретического распределения для распределения данных ROC AUC
#     #data_type = 'roc'
#     #define_distribution.check_for_agr(data_type, np_roc_list)
#     #
#     # dist = define_distribution.get_best_distribution(data_type, roc_train_data, roc_valid_data, roc_test_data, 
#     #                                                   sigma_ratio, simulationAmount)
# 
#     #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     #    pd.set_option("colheader_justify", "left")
#     #    print(f"Наилучшее распределение: {dist}")
# =============================================================================
    
    
    


if __name__ == "__main__":
    # Начало программы
    #  для фиксации значений выставляем seed
    np.random.seed(0)
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    main_process()
