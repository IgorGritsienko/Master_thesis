import numpy as np
import random
import read_parameters

def Generate_CTR(freeCoef, weekCoef, week, noise):                             # генерация КТИ
    return freeCoef + weekCoef * week + noise


def GenerateNoise(deviation):                                                  # генерация одной записи шума
    return random.gauss(0, deviation)


def Create_intervals(filepath):
    frequency = read_parameters.Read_parameters(filepath)
    frequency = [float(x) for x in frequency]

    intervals = []
    for i in range(len(frequency)):
        if (len(intervals) == 0):
            intervals.append(frequency[i])
        else:
            intervals.append(intervals[i-1] + frequency[i])
    return intervals


# генерируем вещественное число от 0 до 100 (вероятность)
# перебираем интервалы, отвечающие за вероятность
# если число больше интервала, то переходим к следующему
# иначе отмечаем, что кол-во записей КТИ для данной недели, соответствующей вероятности,
# увеличивается на 1
def Generate_weeks_frequency(intervals, records_amount, weeks):
    for i in range(records_amount):
        number = random.uniform(0.0, 1.0)
        for j in range(len(intervals)):
            if (number > intervals[j]):
                continue
            else:
                weeks[j][1] = weeks[j][1] + 1
                break  
    return weeks


def Sample_simulation(free_coef, week_coef, deviation, amount, weeks, interval):
    
    noise = []
    for i in range(amount):                                                     # создание массива с шумом
        noise.append(GenerateNoise(deviation))
                
    weeks = Generate_weeks_frequency(interval, amount, weeks)
     
    CTR = []
    
    for i in range(len(weeks)):                                                 # "распакуем" частоту в недель (н-р: 2 записи 11 недели распакуются в "11, 11")
        for j in range(weeks[i][1]):
            CTR.append(weeks[i][0])

    for i in range(amount):                                                     # генерация КТИ
        CTR.append(Generate_CTR(free_coef, week_coef, CTR[i], noise[i]))   # генерируется в один столбец, продолжая записи недель
   
    CTR =  np.array(CTR)                                                        # преобразуем list в numpy array для дальнейшей работы

    return CTR
