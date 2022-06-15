import numpy as np
import copy
from math import sqrt
from math import pi
import random
import files


def Generate_CTR(freeCoef, weekCoef, week, noise):
     # генерация КТИ
    return freeCoef + weekCoef * week + noise

def GenerateNormalNoise(rng, deviation):
    # генерация одной записи шума
    return rng.normal(0, deviation) # gauss

def GenerateLaplaceNoise(rng, deviation):
    b = deviation / sqrt(2)
    return rng.laplace(0, b)

def GenerateUniformnNoise(rng, deviation):
    b = deviation * sqrt(3)
    a = -b
    return rng.uniform(a, b)

def GenerateGumbelNoise(rng, deviation):
    Euler_Mascheroni_constant = 0.57721
    b = sqrt(6) * deviation / pi
    m = -Euler_Mascheroni_constant * b
    return rng.gumbel(m, b)

def GenerateLogisticNoise(rng, deviation):
    s = sqrt(3) * deviation / pi
    return rng.logistic(0, s)

def Create_intervals(filepath):
    """
    Создаем интервалы в соответствии с частотами.
    Интервалы являются накапливаемыми, от 0 до 1 (100).
    """

    frequency = files.Read_parameters(filepath)
    frequency = [float(x) for x in frequency]

    intervals = []
    for i in range(len(frequency)):
        if (len(intervals) == 0):
            intervals.append(frequency[i])
        else:
            intervals.append(intervals[i-1] + frequency[i])
    return intervals


def Generate_weeks_frequency(intervals, records_amount, weeks):
    """
    генерируем вещественное число от 0 до 100 (вероятность)
    перебираем интервалы, отвечающие за вероятность
    если число больше интервала, то переходим к следующему
    иначе отмечаем, что кол-во записей КТИ для данной недели, соответствующей вероятности,
    увеличивается на 1
    """

    for i in range(records_amount):
        number = random.uniform(0.0, 1.0)
        for j in range(len(intervals)):
            if (number > intervals[j]):
                continue
            else:
                weeks[j][1] = weeks[j][1] + 1
                break
    return weeks


def Sample_simulation(distribution, free_coef, week_coef, deviation, amount, weeks, interval):
    noise = []
    amount = int(amount)
    rng = np.random.default_rng(seed=42)
    if (distribution == 'norm'):
        for i in range(amount):                                                     # создание массива с шумом
            noise.append(GenerateNormalNoise(rng, deviation))
    elif (distribution == 'laplace'):
         for i in range(amount):                                                    
            noise.append(GenerateLaplaceNoise(rng, deviation))
    elif (distribution == 'uniform'):
        for i in range(amount):                                                     
            noise.append(GenerateUniformnNoise(rng, deviation))
    elif (distribution == 'logistic'):
        for i in range(amount):                                                     
            noise.append(GenerateLogisticNoise(rng, deviation))
    elif (distribution == 'gumbel'):
        for i in range(amount):                                                     
            noise.append(GenerateGumbelNoise(rng, deviation))


    weeks = Generate_weeks_frequency(interval, amount, weeks)

    CTR = []
    # "распакуем" частоту в недель (н-р: 2 записи 11 недели распакуются в "11, 11")
    for i in range(len(weeks)):
        for j in range(weeks[i][1]):
            CTR.append(weeks[i][0])

    # генерация КТИ
    for i in range(int(amount)):
        # генерируется в один столбец, продолжая записи недель
        CTR.append(Generate_CTR(free_coef, week_coef, CTR[i], noise[i]))
        # преобразуем list в numpy array для дальнейшей работы
    CTR =  np.array(CTR)

    return CTR


def Data_union(norm_data, anem_data, norm_amount, anem_amount):
    """
    Из каждого одномерного массива делаем двумерный с двумя столбцами - недели и КТИ.
    Вертикально соединяем получившиеся массивы.
    """
    norm_data = np.reshape(norm_data, [int(norm_data.size / 2), 2], order="F")
    anem_data = np.reshape(anem_data, [int(anem_data.size / 2), 2], order="F")
    X = np.vstack([norm_data, anem_data])
    return X


def Add_y(norm_amount, anem_amount, X):
    """
    Первые norm_amount элементов генерируются, как '1'.
    Оставшиеся, как '0'.
    1 - нормальные данные.
    0 - выявлена анемия.
    В конце производим склейку нового столбца.
    """

    y = np.zeros(norm_amount)             # создаем вектор принадлежности наблюдений к классам
    y = np.append(y, np.ones(anem_amount)).reshape((-1, 1))
    Xy = np.hstack((X, y))
    return Xy


def manipulate_gen_data(distribution, weeks_norm, weeks_anem, normal_amount, anemia_amount, scaler, filename, intervals_norm, intervals_anem, params):
    """
    Моделируем данные.
    Объединяем нормальный и анемичные данные в один массив.
    Тренируем Скейлер и используем его.
    Добавляем target (0 или 1) к данным.
    Сохраняем в файл.
    """
    
    # копия недель с частотами (изначально частоты по 0)
    tmp_weeks_norm = copy.deepcopy(weeks_norm)
    tmp_weeks_anem = copy.deepcopy(weeks_anem)

    #  моделирование выборки
    normal_data = Sample_simulation(distribution, params[0], params[1], params[2],
                                    normal_amount, tmp_weeks_norm, intervals_norm)
    anemia_data = Sample_simulation(distribution, params[4], params[5], params[6],
                                    anemia_amount, tmp_weeks_anem, intervals_anem)

    #  объединение данных
    X = Data_union(normal_data, anemia_data, normal_amount, anemia_amount)

    #  тренируем скейлер на исходной выборке
    #scaler.fit(X)

    # применяем его к исходной выборке
    #scaled_X = scaler.transform(X)

    # добавляем столбец Y (target) к данным - 0 или 1
    Xy = Add_y(normal_amount, anemia_amount, X) #scaled_X

    #  сохранение сгенерированных данных в файл
    files.Save_sample_to_csv(Xy, filename, _mode = 'w')
    
    return Xy
