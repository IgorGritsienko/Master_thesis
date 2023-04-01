import copy
import random
from dataclasses import dataclass
from math import pi
from math import sqrt

import numpy as np

import files

EULER_MASCHERONI_NUMBER = 0.57721
NORMAL_DISTRIBUTION = 'norm'
LAPLACE_DISTRIBUTION = 'laplace'
UNIFORM_DISTRIBUTION = 'uniform'
LOGISTIC_DISTRIBUTION = 'logistic'
GUMBEL_DISTRIBUTION = 'gumbel'


@dataclass
class Noise:
    std: float

    def generate_normal_noise(self, rng):
        # генерация одной записи шума
        return rng.normal(0, self.std)  # gauss

    def generate_laplace_noise(self, rng):
        b = self.std / sqrt(2)
        return rng.laplace(0, b)

    def generate_uniform_noise(self, rng):
        b = self.std * sqrt(3)
        a = -b
        return rng.uniform(a, b)

    def generate_logistic_noise(self, rng):
        s = sqrt(3) * self.std / pi
        return rng.logistic(0, s)

    def generate_gumbel_noise(self, rng):
        b = sqrt(6) * self.std / pi
        m = -EULER_MASCHERONI_NUMBER * b
        return rng.gumbel(m, b)


def create_intervals(filepath):
    """
    Создаем интервалы в соответствии с частотами.
    Интервалы являются накапливаемыми, от 0 до 1 (100).
    """
    frequency = files.read_parameters(filepath)
    frequency = [float(x) for x in frequency]

    intervals = []
    for i in range(len(frequency)):
        if len(intervals) == 0:
            intervals.append(frequency[i])
        else:
            intervals.append(intervals[i - 1] + frequency[i])
    return intervals


def generate_weeks_frequency(weeks, intervals, records_amount):
    """
    Генерируем вещественное число от 0 до 100 (вероятность).
    Перебираем интервалы, отвечающие за вероятность.
    Если число больше интервала, то переходим к следующему,
    иначе отмечаем, что кол-во записей КТИ для данной недели, соответствующей вероятности,
    увеличивается на 1
    """
    for i in range(records_amount):
        number = random.uniform(0.0, 1.0)
        for j in range(len(intervals)):
            if number > intervals[j]:
                continue
            else:
                weeks[j][1] = weeks[j][1] + 1
                break
    return weeks


def sample_simulation(distribution, model_ctr, weeks, interval, amount):
    """

    """
    noise_list = []
    rng = np.random.default_rng(seed=42)
    noise = Noise(model_ctr.std)
    if distribution == NORMAL_DISTRIBUTION:
        for i in range(amount):  # создание массива с шумом
            noise_list.append(noise.generate_normal_noise(rng))
    elif distribution == LAPLACE_DISTRIBUTION:
        for i in range(amount):
            noise_list.append(noise.generate_laplace_noise(rng))
    elif distribution == UNIFORM_DISTRIBUTION:
        for i in range(amount):
            noise_list.append(noise.generate_uniform_noise(rng))
    elif distribution == LOGISTIC_DISTRIBUTION:
        for i in range(amount):
            noise_list.append(noise.generate_logistic_noise(rng))
    elif distribution == GUMBEL_DISTRIBUTION:
        for i in range(amount):
            noise_list.append(noise.generate_gumbel_noise(rng))

    weeks_frequency = generate_weeks_frequency(weeks, interval, amount)

    weeks_list = []
    ctr_index_list = []
    # "распаковываем" частоту в недель (н-р: 2 записи 11-й недели распакуются в "11, 11")
    for i in range(len(weeks_frequency)):
        for j in range(weeks_frequency[i][1]):
            weeks_list.append(weeks_frequency[i][0])

    # генерация КТИ
    for i in range(int(amount)):
        # генерируется в один столбец, продолжая записи недель
        ctr_index_list.append(model_ctr.calculate_ctr_index(weeks_list[i], noise_list[i]))

    # преобразуем list в numpy array для дальнейшей работы
    weeks_list = np.array(weeks_list).reshape(-1, 1)
    ctr_index_list = np.array(ctr_index_list).reshape(-1, 1)

    x = np.concatenate((weeks_list, ctr_index_list), axis=1)

    return x


def add_y(x, normal_data_amount, anemia_data_amount):
    """
    Первые norm_amount элементов генерируются, как '0'.
    Оставшиеся, как '0'.
    0 - нормальные данные.
    1 - выявленная анемия.
    В конце производим склейку нового столбца.
    """
    y = np.zeros(normal_data_amount)  # создаем вектор принадлежности наблюдений к классам
    y = np.append(y, np.ones(anemia_data_amount)).reshape((-1, 1))
    xy = np.hstack((x, y))

    return xy


def generate_sample(distribution,
                    normal_model_ctr, anemia_model_ctr,
                    weeks_normal, weeks_anemia,
                    intervals_normal, intervals_anemia,
                    normal_data_amount, anemia_data_amount,
                    generated_sample_file):
    """
    Моделируем данные.
    Объединяем нормальный и анемичные данные в один массив.
    Тренируем Скейлер и используем его.
    Добавляем target (0 или 1) к данным.
    Сохраняем в файл.
    """
    # копия недель с частотами (изначально частоты по 0)
    tmp_weeks_normal = copy.deepcopy(weeks_normal)
    tmp_weeks_anemia = copy.deepcopy(weeks_anemia)

    #  моделирование выборки
    x_normal = sample_simulation(distribution,
                                 normal_model_ctr,
                                 tmp_weeks_normal,
                                 intervals_normal,
                                 normal_data_amount)
    x_anemia = sample_simulation(distribution,
                                 anemia_model_ctr,
                                 tmp_weeks_anemia,
                                 intervals_anemia,
                                 anemia_data_amount)
    # объединить нормальные и анемичные наблюдения
    # столбец: сначала нормальные, далее - анемичные
    x = np.vstack([x_normal, x_anemia])
    xy = add_y(x, normal_data_amount, anemia_data_amount)

    # сохранение сгенерированных данных в файл
    # mode 'a' - добавление новых записей к старым, создание файла при его отсутствии
    # 'w' - перезапись
    files.save_sample_to_csv(xy, generated_sample_file, save_mode='w')

    return xy
