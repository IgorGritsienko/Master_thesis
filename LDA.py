import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import copy
import math
import statistics

import define_distribution
import generation
import read_parameters

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names



def Save_sample_to_csv(array, filename, _mode):
    df = pd.DataFrame(data={"Weeks": array[:, 0], "CTR": array[:, 1], "target": array[:, 2]})       # 3 столбца датафрейма: КТИ, недели и класс (анемия или норма)
    #df = pd.DataFrame(np.array([array[:, 0], array[:, 1], array[:, 2]]).T)
    df.to_csv(filename, sep=';',index=False, mode=_mode, header=False, float_format='%.6f')              # mode 'a' - добавление новых записей к старым, создание файла при его отсутствии. 'w' - перезапись

                                                                                                  # sep=',' для записи в один столбец через ','
def Read_csv(filename, n_rows, position):
    data_frame = pd.read_csv(filename, nrows=n_rows, skiprows=position, sep=';', header=0)
    return data_frame



                                                                   
def Scatter_plot(X, Y, coef1, coef2, const):                                   # график рассеивания 
    x = np.array([-2, 2])                                                      # ось Х 
    y_hyperplane = -(const + x * coef1) / coef2                                # построение дискриминирующей линии 
    
    plt.figure()
    colors = ['turquoise', 'darkorange'] 
    target_names = ['anemia', 'normal']
    
    plt.plot(x, y_hyperplane, 'k')
    
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], alpha=.8, color=color, label=target_name)   # ОХ - недели, ОУ - КТИ
    plt.legend(loc='best', shadow=False)
    plt.title('The scattering of observations by features')    
    plt.xlabel('Weeks')
    plt.ylabel('CTR')
    plt.show()
    

def Data_union(norm_data, anem_data, norm_amount, anem_amount):
                           
    norm_data = np.reshape(norm_data, [int(norm_data.size / 2), 2], order="F")               # делаем 2 столбца: недели и кти
    anem_data = np.reshape(anem_data, [int(anem_data.size / 2), 2], order="F")
    X = np.vstack([norm_data, anem_data])                                                    # соединяем записи   
    return X

def Add_y(norm_amount, anem_amount, X):
    y = np.ones(norm_amount)                                                               # создаем вектор принадлежности наблюдений к классам 
    y = np.append(y, np.zeros(anem_amount)).reshape((-1, 1))  
    Xy = np.hstack((X, y))
    return Xy

def Transform_Y(y):                                                                 
    y_new = []
    for el in y:
        if (el == 0):
            y_new.append([1, 0])
        else:
            y_new.append([0, 1])
    return np.reshape(y_new, [np.size(y_new, 0), -1])

def Compute_ROC_curve_ROC_area(Y_test, Y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(Y_test, Y_score)
    roc_auc = auc(fpr, tpr)
        
    return fpr, tpr, roc_auc

def Plot_ROC(fpr_, tpr_, roc_auc_):
    plt.figure()
    lw = 2
    plt.plot(fpr_, tpr_, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()    


# Начало программы
#np.random.seed(1)
warnings.filterwarnings("ignore")
_input = './input.txt'
records = './records.csv'
dif_test_records = './dif_test_records.csv'
train_sample = './train_sample.csv'
test_sample = './test_sample.csv'
params = './params.txt'
frequency_norm = './frequency_norm.txt'
frequency_anem = './frequency_anem.txt'

if os.path.exists(records):                                                    # если существует файл с записями, очистить его
    f = open(records, 'r+')
    f.truncate(0)
    f.close()

model_parameters = read_parameters.Read_parameters(_input)

freeCoefNormal = float(model_parameters[0])
weekCoefNormal = float(model_parameters[1])
deviationNormal = float(model_parameters[2])   
normalAmount = int(model_parameters[3])
freeCoefAnemia = float(model_parameters[4])
weekCoefAnemia = float(model_parameters[5])
deviationAnemia = float(model_parameters[6])
anemiaAmount = int(model_parameters[7])

parameters = read_parameters.Read_parameters(params)

simulationAmount = int(parameters[0])                                   # количество запусков программы
isTestSampleCustom = int(parameters[1])                                 # будет ли отдельная тестовая выборка со своими параметрами
N_train_rate = float(parameters[2])                                          # доля обучающей выборки

N_train_records = int((normalAmount + anemiaAmount) * N_train_rate)          # размер обучающей выборки


N = normalAmount + anemiaAmount                                                 # количество всех записей

test_normal_amount = round(normalAmount * (1 - N_train_rate))               # объем записей для теста (в случае использования тест. выборки с другими параметрами)
test_anemia_amount = round(anemiaAmount * (1 - N_train_rate))


intervals_norm = generation.Create_intervals(frequency_norm)                               # создали интервалы с вероятностными частотами
intervals_anem = generation.Create_intervals(frequency_anem)

# анемия с 15 недели
# нормальные с 11 недели
weeks_norm = range(11, 35)                                                     # создали недели, на которых могут проводиться измерения
weeks_norm = [[x] + [0] for x in weeks_norm]                                   # делаем каждому элементу пару - будущее количество измероений на данной неделе 
weeks_anem = range(15, 35) 
weeks_anem = [[x] + [0] for x in weeks_anem]


spec_list = []
sens_list = []
roc_list = []

scaler = preprocessing.StandardScaler()

for i in range(simulationAmount):   
    
    tmp_weeks_norm = copy.deepcopy(weeks_norm)                                  # копия недель с частотами (изначально частоты по 0)
    tmp_weeks_anem = copy.deepcopy(weeks_anem)
    
    normal_data = generation.Sample_simulation(freeCoefNormal, weekCoefNormal, deviationNormal, normalAmount, tmp_weeks_norm, intervals_norm)  # моделирование выборки
    anemia_data = generation.Sample_simulation(freeCoefAnemia, weekCoefAnemia, deviationAnemia, anemiaAmount, tmp_weeks_anem, intervals_anem)
    
    X = Data_union(normal_data, anemia_data, normalAmount, anemiaAmount)
    
    scaler.fit(X)                                                               # тренируем скейлер на исходной выборке   
    
    scaled_X = scaler.transform(X)                                              # применяем его к исходной выборке
    
    Xy = Add_y(normalAmount, anemiaAmount, scaled_X)                            # добавляем столбец Y к данным
    
    Save_sample_to_csv(Xy, records, _mode = 'a')
    
    if (isTestSampleCustom == 1):
        
        if os.path.exists(dif_test_records):                                                            # очистить файл, если он существует
            f = open(dif_test_records, 'r+')
            f.truncate(0)
            f.close()
    
    
        # моделирование тестовой выборки с другими параметрами    
        tmp_weeks_norm = copy.deepcopy(weeks_norm)
        tmp_weeks_anem = copy.deepcopy(weeks_anem)
        
        test_normal_data = generation.Sample_simulation(freeCoefNormal, weekCoefNormal, deviationNormal, test_normal_amount, tmp_weeks_norm, intervals_norm)
        test_anemia_data = generation.Sample_simulation(freeCoefAnemia, weekCoefAnemia, deviationAnemia, test_anemia_amount, tmp_weeks_anem, intervals_anem)
        
        X1 = Data_union(test_normal_data, test_anemia_data, test_normal_amount, test_anemia_amount)
                                
        scaled_X1 = scaler.transform(X1)
        
        Xy1 = Add_y(test_normal_amount, test_anemia_amount, scaled_X1)
    
        Save_sample_to_csv(Xy1, dif_test_records, _mode = 'a')

        


prob_norm = 0.1
#while (prob_norm <= 0.95):    
position = 0                                                                   # позиция, с которой необходимо считывать записи для текущего эксперимента.
                                                                               # каждый раз увеличивается на N
position_shift_dif_test = test_normal_amount + test_anemia_amount  
                                                                           
for i in range(simulationAmount):
    Xy = Read_csv(records, N, position)

    position += N
    Xy = Xy.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(Xy[:, :2], Xy[:, 2], test_size = (1 - N_train_rate), random_state=np.random.randint(10000))
    

    if (isTestSampleCustom == 1):                                                                 # если для теста используется другая выборка
        position_dif_test = 0                                                                     # начальная позиция считывания
        Xy_test = Read_csv(dif_test_records, int(N * N_train_rate), position_dif_test)            # генерируем, делаем шаг и считываем количество записей, равное заданному объему тестовой выборки
        position_dif_test += position_shift_dif_test                                              # каждый запуск сдвигается позиция считывания                                             
        Xy_test = Xy_test.to_numpy()
        test_amount = int(N * (1 - N_train_rate))                                                 # определить необходимое число записей для теста
        X_test = np.array(Xy_test[:test_amount, :2])                                              # берем только часть выборки, объем которой = test_amount                                  
        y_test = np.array(Xy_test[:test_amount, 2])

      
    #priors1 = np.bincount(y_train.astype(int)) / len(y_train)                 # априорные вероятности (подсчитываются внутри LDA, если не заданы) 
                                                                               # считает вхождение каждого уникального значения (1 и 0) и делит на все кол-во для получения долей     
    prob_anem = 1 - prob_norm
                                                         
    clf = LinearDiscriminantAnalysis(priors=[prob_norm, prob_anem])                      # priors - аргумент. устанавляиваются две приорные вероятности так, чтобы сумма была равна 1
    clf.fit(X_train, y_train)                                                  # тренируем 
    y_predict = clf.predict(X_test)                                            # предсказываем
     
     
    
    #print("Коэффициенты: ", clf.coef_)                                                        # коэффициенты дискр. ф-и
    #print("Константа: ", clf.intercept_)                                                      # константа
     
    #precision = precision_score(y_test, Y_predict)
     
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
     
    specificity = tn / (tn + fp)                                               # специфичность       | сколько здоровых было определено правильно из числа всех здоровых
    recall = recall_score(y_test, y_predict)                                   # чувствительность    | сколько больных было определено правильно из числа всех больных
     
    #print("Специфичность: ", specificity)
    #print("Чувствительность: ", recall)
    #print(confusion_matrix(y_test, y_predict))
     
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    #Scatter_plot(X_test, y_test, clf.coef_[0][0], clf.coef_[0][1], clf.intercept_)
     
    #y_train = Transform_Y(y_train)
    #y_score = clf.fit(X_train, y_train).decision_function(X_test)
    
    roc_auc = roc_auc_score(y_test, clf.decision_function(X_test))
    y_score = clf.decision_function(X_test)
    #fpr, tpr, roc_auc = Compute_ROC_curve_ROC_area(y_test, y_score)            # The x-axis showing 1 – specificity (= false positive fraction = FP/(FP+TN))
                                                                        # The y-axis showing sensitivity (= true positive fraction = TP/(TP+FN)) 
     
    #Plot_ROC(fpr, tpr, roc_auc)
     
    #y_train = np.reshape(y_train, [np.size(y_train), 1])                       # сохранение данных, на которых проводится тренировка и тест 
    #Save_sample_to_csv(np.hstack((X_train, y_train)), train_sample, _mode = 'w')
    #y_test = np.reshape(y_test, [np.size(y_test), 1])
    #Save_sample_to_csv(np.hstack((X_test, y_test)), test_sample, _mode = 'w')
     
    #coefs = np.reshape(np.array([clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0]]), [-1, 3])
 
    spec_list.append(specificity)
    sens_list.append(recall)
    roc_list.append(roc_auc)

 
 #Save_sample_to_csv(coefs, './coefs.txt', _mode = 'w')


np_spec_list = np.array(spec_list)
np_sens_list = np.array(sens_list)
np_roc_list = np.array(roc_list)

np.savetxt('./spec_results.txt', np_spec_list)
np.savetxt('./sens_results.txt', np_sens_list)
np.savetxt('./roc_results.txt', np_roc_list)

#print('{0:.3f}'.format(np.mean(np_spec_list)))
print("Probabilities: norm: {0:.2f}, anem: {1:.2f}".format(prob_norm, prob_anem))
print(np.around(np.mean(np_spec_list), 3))
print(np.around(np.std(np_spec_list), 3))

print(np.around(np.mean(np_sens_list), 3))
print(np.around(np.std(np_sens_list), 3))

print(np.around(np.mean(roc_list), 3))
print(np.around(np.std(roc_list), 3))
print("\n |-------------------------------------------| \n")
#prob_norm = prob_norm + 0.1 # для цикла while при переборе априорных вероятностей


#if(prob_norm == 0.8):
# =============================================================================
# fig, axs = plt.subplots(3, 1, sharey=True, sharex=True)
#  
# axs[0].hist(spec_list)
# axs[0].set_xlabel('Специфичность')
# 
# axs[1].hist(sens_list)
# axs[1].set_xlabel('Чувствительность')
#  
# axs[2].hist(roc_list)
# axs[2].set_xlabel('AUC')
# 
# 
# =============================================================================

np.random.shuffle(np_spec_list) #DevSkim: ignore DS148264
length = int(np_spec_list.size / 2)
np_spec_list_train, np_spec_list_test = np_spec_list[:length], np_spec_list[length:]

number_distributions_to_plot = 1
dist, params = define_distribution.get_best_distribution(np_spec_list_train, number_distributions_to_plot)
D, p = st.kstest(np_spec_list_test, dist, args=params)



print(dist, D, p)
